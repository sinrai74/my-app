"""正式G2：C-1 の構造確認（合成した program dict。ネットワークは使わない）。

用途の制限：入力の構造の確認だけ。実データの再現・正式G2 の証拠ではない。
実際の出走表の取得経路（過去の日付・再現性・項目の網羅性）は Q8 で確かめる。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import codecs
import hashlib
import json
import os
import sys
import unittest

import requests

ROOT = os.path.realpath(os.getcwd())
DATE = "20990101"


def _boat(lane):
    return {
        "racer_boat_number": lane,
        "racer_number": 9000 + lane,
        "racer_name": f"合成{lane}",
        "racer_national_top_1_percent": 5.0 + lane / 10,
        "racer_local_top_1_percent": 4.0,
        "racer_assigned_motor_top_2_percent": 30.0,
        "racer_average_start_timing": 0.16,
        "racer_class_number": 2,
    }


def _program(venue, rno, closed):
    return {
        "race_date": "2099-01-01",
        "race_stadium_number": venue,
        "race_number": rno,
        "race_closed_at": closed,
        "race_grade_number": 5,
        "boats": [_boat(i) for i in range(1, 7)],
    }


PROGRAMS = {"programs": [
    _program(3, 2, "2099-01-01 11:40:00"),
    _program(1, 1, "2099-01-01 11:40:00"),
    _program(2, 1, "2099-01-01 10:50:00"),
]}
RAW = json.dumps(PROGRAMS, ensure_ascii=False).encode("utf-8")


def fake_response(content, status=200):
    """requests.Response を作り、content を直接与える（ネットワークは使わない）。"""
    r = requests.models.Response()
    r.status_code = status
    r._content = content
    r.encoding = None
    r.url = "https://example.invalid/"
    return r


class FixProgramsTests(unittest.TestCase):
    def test_fixed_once_hashed_and_ordered(self):
        from formal_g2.programs import fix_programs

        calls = []

        def get(url):
            calls.append(url)
            return fake_response(RAW)

        fixed = fix_programs(DATE, http_get=get)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith(f"/2099/{DATE}.json"))
        self.assertEqual(fixed["raw_sha256"], hashlib.sha256(RAW).hexdigest())
        self.assertEqual([t["eval_id"] for t in fixed["targets"]],
                         [f"{DATE}_02_01", f"{DATE}_01_01", f"{DATE}_03_02"])

    def test_bom_is_parsed_like_response_json(self):
        from formal_g2.programs import fix_programs

        bom = codecs.BOM_UTF8 + RAW
        fixed = fix_programs(DATE, http_get=lambda u: fake_response(bom))
        self.assertEqual(len(fixed["targets"]), 3)
        self.assertEqual(fixed["raw_sha256"], hashlib.sha256(bom).hexdigest())

    def test_unavailable_cases_match_safe_get_contract(self):
        from formal_g2.programs import ProgramsUnavailable, fix_programs

        with self.assertRaises(ProgramsUnavailable):  # 空
            fix_programs(DATE, http_get=lambda u: fake_response(b'{"programs": []}'))
        with self.assertRaises(ProgramsUnavailable):  # HTTP エラー（raise_for_status）
            fix_programs(DATE, http_get=lambda u: fake_response(b"not found", status=404))
        with self.assertRaises(ProgramsUnavailable):  # JSON を読めない（RequestException の一種）
            fix_programs(DATE, http_get=lambda u: fake_response(b"<html>"))

        def conn_error(url):
            raise requests.ConnectionError("network")

        with self.assertRaises(ProgramsUnavailable):
            fix_programs(DATE, http_get=conn_error)

        def other_error(url):
            raise RuntimeError("unexpected")

        with self.assertRaises(RuntimeError):  # RequestException 以外は外へ伝える
            fix_programs(DATE, http_get=other_error)

    def test_bad_date_rejected(self):
        from formal_g2.programs import fix_programs

        with self.assertRaises(ValueError):
            fix_programs("2099-01-01", http_get=lambda u: fake_response(RAW))


class VenueStatsValidationTests(unittest.TestCase):
    def test_validate(self):
        from formal_g2.race_eval import validate_venue_stats

        good = {"全国平均": {lane: {"samples": 10} for lane in range(1, 7)}, "桐生": {}}
        self.assertEqual(validate_venue_stats(good), [])
        self.assertTrue(validate_venue_stats({}))
        bad_lane = {"全国平均": {lane: {"samples": 10} for lane in range(1, 6)}}
        self.assertIn("national lane 6 missing", validate_venue_stats(bad_lane))
        zero = {"全国平均": {lane: {"samples": 0 if lane == 3 else 5} for lane in range(1, 7)}}
        self.assertIn("national lane 3 samples <= 0", validate_venue_stats(zero))


class ProviderInjectionTests(unittest.TestCase):
    def test_one_fetcher_call_per_race_preload_match_and_records(self):
        from formal_g2 import race_eval
        from formal_g2.instrument import Instrument
        from formal_g2.programs import fix_programs
        from x_asahi_scoring import load_asahi_config

        fixed = fix_programs(DATE, http_get=lambda u: fake_response(RAW))
        inst = Instrument(ROOT)
        inst.start()
        try:
            preload = race_eval.preload_venue_caches(ROOT, DATE)
            ctx = race_eval.build_context(ROOT, DATE, fixed["programs"], load_asahi_config(),
                                          race_eval.load_comparator(ROOT), inst, preload)
            recs = [race_eval.evaluate_race(ctx, t) for t in fixed["targets"]]
            missing = race_eval.evaluate_race(ctx, {"eval_id": f"{DATE}_09_09", "venue_num": 9,
                                                    "race_number": 9})
        finally:
            inst.stop()
        self.assertTrue(preload["ok"], preload.get("problems"))
        for rec in recs:
            self.assertIsNone(rec.get("exception"), rec.get("exception"))
            self.assertEqual(rec["counts"]["fetcher_calls"], 1)
            self.assertEqual(rec["counts"]["BoatsProvider.resolve_boats"], 1)
            self.assertTrue(rec["d13_2"]["venue_caches_match_preload"], rec["preload_match"])
            self.assertTrue(rec["provider_cache_read_check"]["unchanged"])
            self.assertIn("d5", rec)
            for label in ("score_by_model", "_load_ml_model", "_predict_win_prob"):  # W7-b
                self.assertEqual(rec["counts"][label], 0, label)
        self.assertEqual(missing["result_category"], "not_established")
        # ルートの features.py は実行も読み込みもされない
        self.assertFalse(inst.module_flags["root_features"])
        self.assertFalse(inst.module_flags["config"])
        root_features = os.path.join(ROOT, "features.py")
        loaded = [n for n, m in list(sys.modules.items())
                  if os.path.realpath(getattr(m, "__file__", "") or "") == root_features]
        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
