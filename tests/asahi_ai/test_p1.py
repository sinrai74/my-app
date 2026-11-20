"""朝刊AI 新モデル P1（データの土台）の試験。合成ファイルで、仕様（Project 05 第3版）の各規則を確かめる。"""
from __future__ import annotations

import datetime as dt
import struct
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from asahi_ai import archive, build, codes, fetch, parse_b, parse_k

from . import synth

D = dt.date(2023, 1, 1)
COND = "晴　  風  北東　 5m  波　  5cm"


def levels(checks, check):
    return [c["level"] for c in checks if c["check"] == check]


class TestCodes(unittest.TestCase):
    def test_result_classes(self):
        self.assertEqual(codes.classify_result("03")["finish_position"], 3)
        for c, cls, resp in (("F", "flying", "responsible"), ("L0", "late_start", "not_responsible"),
                             ("L1", "late_start", "responsible"), ("K0", "absent", "not_responsible"),
                             ("K1", "absent", "responsible"), ("S0", "disqualified", "not_responsible"),
                             ("S2", "disqualified", "responsible")):
            r = codes.classify_result(c)
            self.assertEqual((r["finish_position"], r["result_class"], r["responsibility"]), (None, cls, resp), c)

    def test_unknown_code_is_not_converted(self):
        for c in ("00", "X9", "07"):
            r = codes.classify_result(c)
            self.assertEqual(r["result_class"], "unknown")
            self.assertIsNone(r["finish_position"])

    def test_st(self):
        self.assertEqual(codes.parse_st("0.07"), ("", 0.07, True))
        self.assertEqual(codes.parse_st("F0.01"), ("F", -0.01, True))   # フライングは負
        self.assertEqual(codes.parse_st("L1.99"), ("L", 1.99, True))   # 出遅れの値は残す
        self.assertEqual(codes.parse_st("L ."), ("L", None, True))
        self.assertEqual(codes.parse_st("K ."), ("K", None, True))
        self.assertEqual(codes.parse_st("abc")[2], False)

    def test_times(self):
        self.assertEqual(codes.parse_race_time("1.48.9"), (108.9, True))
        self.assertEqual(codes.parse_race_time(".  ."), (None, True))
        self.assertEqual(codes.parse_race_time("1.4x.9")[1], False)
        self.assertEqual(codes.parse_exhibition("6.86"), (6.86, True))
        self.assertEqual(codes.parse_exhibition("K ."), (None, True))


class TestArchive(unittest.TestCase):
    def test_crc16_vector(self):
        self.assertEqual(archive.crc16(b"123456789"), 0xBB3D)  # CRC-16/ARC の検査値

    def _fake_lzh(self, crc: int, name=b"K230101.TXT", method=b"-lh5-") -> bytes:
        head = struct.pack("<BB5sII4sBBB", 0, 0, method, 0, 0, b"\0\0\0\0", 0x20, 0, len(name))
        return head + name + struct.pack("<H", crc) + b"\0" * 8

    def test_crc_mismatch_is_rejected(self):
        body = b"hello"
        good = self._fake_lzh(archive.crc16(body))
        bad = self._fake_lzh(archive.crc16(body) ^ 1)
        with mock.patch.object(archive.lzh_extract, "extract_all", return_value={"K230101.TXT": body}):
            self.assertEqual(archive.extract_single(good).data, body)
            with self.assertRaises(archive.ArchiveError):
                archive.extract_single(bad)

    def test_method_and_member_count(self):
        with self.assertRaises(archive.ArchiveError):
            archive.extract_single(self._fake_lzh(0, method=b"-lh1-"))
        with mock.patch.object(archive.lzh_extract, "extract_all", return_value={"a": b"", "b": b""}):
            with self.assertRaises(archive.ArchiveError):
                archive.extract_single(self._fake_lzh(0))


class TestParseB(unittest.TestCase):
    def test_basic_and_three_digit_boat(self):
        racers = synth.std_b_racers()
        racers[5] = synth.b_racer(6, racer_no="4006", motor_2r="44.83", boat_no="158", boat_2r="36.99")
        self.assertIn(b"44.83158 36.99", racers[5])  # 旧パーサが落とした形（空白なしでつながる）
        p = parse_b.parse_b(synth.b_file(races=[(1, racers)]), D)
        self.assertEqual(p.issues, [])
        self.assertEqual(len(p.entries), 6)
        e6 = p.entries[5]
        self.assertEqual((e6["motor_2r"], e6["boat_no"], e6["boat_2r"]), (44.83, 158, 36.99))
        r = p.races[0]
        self.assertEqual((r["race_id"], r["deadline_time"], r["distance_m"], r["meet_day"]),
                         ("202301010901", "15:20", 1800, 4))
        self.assertEqual(p.venues, ["09"])  # 1文字の場名（津）

    def test_first_day_label(self):
        p = parse_b.parse_b(synth.b_file(day_label="初　日", races=[(1, synth.std_b_racers())]), D)
        self.assertEqual(p.races[0]["meet_day"], 1)

    def test_wrong_length_line_stops(self):
        racers = synth.std_b_racers()
        racers[2] = racers[2] + b" "
        p = parse_b.parse_b(synth.b_file(races=[(1, racers)]), D)
        self.assertIn("stop", [i["level"] for i in p.issues if i["check"] == "b_racer_len"])
        self.assertEqual(len(p.entries), 5)

    def test_header_date_mismatch_stops(self):
        p = parse_b.parse_b(synth.b_file(day=2, races=[(1, synth.std_b_racers())]), D)
        self.assertIn("b_header_date", [i["check"] for i in p.issues])

    def test_unreadable_race_header_stops(self):
        raw = synth.b_file(races=[(1, synth.std_b_racers()), (2, synth.std_b_racers())])
        raw = raw.replace("電話投票締切予定１５：２０".encode("cp932"), "締切未定".encode("cp932"), 1)
        p = parse_b.parse_b(raw, D)
        self.assertIn(("b_race_header", "stop"), [(i["check"], i["level"]) for i in p.issues])
        self.assertTrue(all(e["race_id"].endswith("02") for e in p.entries))  # 前のレースに混ぜない

    def test_year_line_missing_falls_back_to_month_day(self):
        raw = synth.b_file(races=[(1, synth.std_b_racers())])
        lines = raw.split(b"\r\n")
        lines = [l for l in lines if "２０２３年".encode("cp932") not in l]
        p = parse_b.parse_b(b"\r\n".join(lines), D)
        self.assertEqual([(i["check"], i["level"]) for i in p.issues], [("b_header_year_missing", "warn")])
        p2 = parse_b.parse_b(b"\r\n".join(lines), dt.date(2023, 1, 2))
        self.assertIn(("b_header_date", "stop"), [(i["check"], i["level"]) for i in p2.issues])

    def test_placeholder_venue_warns(self):
        raw = b"\r\n".join([b"STARTB", b"11BBGN", "ボートレースびわこ".encode("cp932"),
                             "この場のデータ更新は、いましばらくお待ちください。".encode("cp932"), b"11BEND", b"FINALB", b""])
        p = parse_b.parse_b(raw, D)
        self.assertEqual([(i["check"], i["level"]) for i in p.issues], [("b_venue_placeholder", "warn")])
        rawk = b"\r\n".join([b"STARTK", b"11KBGN", "ボートレースびわこ".encode("cp932"),
                              "データは、この場の全レース終了後に登録されます。".encode("cp932"), b"11KEND", b"FINALK", b""])
        pk = parse_k.parse_k(rawk, D)
        self.assertEqual([(i["check"], i["level"]) for i in pk.issues], [("k_venue_placeholder", "warn")])

    def test_bad_value_quarantined(self):
        racers = synth.std_b_racers()
        racers[0] = synth.b_racer(1, racer_no="4001", cls="Z9")
        p = parse_b.parse_b(synth.b_file(races=[(1, racers)]), D)
        self.assertEqual([(i["check"], i["level"]) for i in p.issues], [("b_value", "quarantined")])


class TestParseK(unittest.TestCase):
    def k(self, racers, pays=None, cond=COND, tech="逃げ　　　", summary=((1, "1-2-3", 600),)):
        return parse_k.parse_k(synth.k_file(races=[(1, racers, pays if pays is not None else synth.std_pays(),
                                                     cond, tech)], summary=summary), D)

    def test_basic(self):
        p = self.k(synth.std_k_racers())
        self.assertEqual(p.issues, [])
        self.assertEqual(len(p.results), 6)
        r = p.races[0]
        self.assertEqual((r["weather"], r["wind_dir"], r["wind_speed_m"], r["wave_cm"], r["winning_technique"]),
                         ("晴", "北東", 5, 5, "逃げ"))
        kinds = sorted((x["bet_type"], x["combination"]) for x in p.payouts)
        self.assertIn(("wide", "1-3"), kinds)  # 拡連複の続きの行
        self.assertEqual(len(p.payouts), 10)
        self.assertEqual(p.summary_trifecta["202301010901"], ("1-2-3", 600))

    def test_flying_late_absent(self):
        racers = synth.std_k_racers(codes={4: "F", 5: "L1", 6: "K0"},
                                    sts={4: "F0.01", 5: "L1.99", 6: "K ."})
        racers[5] = synth.k_racer("K0", 6, racer_no="4006", course="", ex="K .", st="K .", rt=".  .")
        p = self.k(racers)
        by = {r["lane"]: r for r in p.results}
        self.assertEqual((by[4]["st_flag"], by[4]["st_value"], by[4]["result_class"]), ("F", -0.01, "flying"))
        self.assertEqual((by[5]["st_flag"], by[5]["st_value"]), ("L", 1.99))
        self.assertEqual((by[6]["result_class"], by[6]["course"], by[6]["exhibition_time"]), ("absent", None, None))
        self.assertEqual(p.issues, [])

    def test_unknown_code_quarantined(self):
        p = self.k(synth.std_k_racers(codes={6: "00"}))
        self.assertEqual([(i["check"], i["level"]) for i in p.issues], [("k_unknown_code", "quarantined")])
        self.assertEqual([r["result_code_raw"] for r in p.results if r["lane"] == 6], ["00"])

    def test_special_payout(self):
        pays = synth.std_pays()
        pays[0] = "        単勝        特払い   70          "
        p = self.k(synth.std_k_racers(), pays)
        sp = [x for x in p.payouts if x["payout_kind"] == "特払い"]
        self.assertEqual((sp[0]["bet_type"], sp[0]["payout_yen"], sp[0]["combination"]), ("win", 70, ""))

    def test_unparsable_payout_quarantined(self):
        pays = synth.std_pays() + ["        ３連単   ？？？"]
        p = self.k(synth.std_k_racers(), pays)
        self.assertIn(("k_payout_line", "quarantined"), [(i["check"], i["level"]) for i in p.issues])

    def test_payout_line_with_leftover_text_quarantined(self):
        pays = synth.std_pays()
        pays[7] = "        ３連単   1-2-3      600  人気     1  不明"
        p = self.k(synth.std_k_racers(), pays)
        self.assertIn(("k_payout_line", "quarantined"), [(i["check"], i["level"]) for i in p.issues])

    def test_unreadable_race_header_stops(self):
        raw = synth.k_file(races=[(1, synth.std_k_racers(), synth.std_pays(), COND, "逃げ")])
        raw = raw.replace(b"H1800m", b"X1800m")
        p = parse_k.parse_k(raw, D)
        self.assertIn(("k_race_header", "stop"), [(i["check"], i["level"]) for i in p.issues])
        self.assertEqual(p.results, [])

    def test_first_day_header(self):
        raw = synth.k_file(races=[(1, synth.std_k_racers(), synth.std_pays(), COND, "逃げ")])
        raw = raw.replace("   第 4日          2023".encode("cp932"), "   初 日          2023".encode("cp932"))
        p = parse_k.parse_k(raw, D)
        self.assertEqual(p.issues, [])

    def test_summary_with_void_and_cancel_rows(self):
        raw = synth.k_file(races=[(1, synth.std_k_racers(), synth.std_pays(), COND, "逃げ")],
                           summary=[(1, "1-2-3", 600)])
        extra = ("\r\n           2R  不成立           不成立           不成立         不成立"
                 "\r\n           3R  中 止"
                 "\r\n           4R  1-2-3     700    1-2-3     100    1-2     100    1-2     100").encode("cp932")
        anchor = "  1-2     100    1-2     100".encode("cp932")
        i = raw.index(anchor) + len(anchor)
        p = parse_k.parse_k(raw[:i] + extra + raw[i:], D)
        self.assertEqual([x for x in p.issues if x["level"] == "stop"], [])
        self.assertEqual(p.summary_trifecta.get("202301010904"), ("1-2-3", 700))
        self.assertEqual(p.cancelled, {"202301010903": "3R  中 止"})  # 要約の表の中止も記録する

    def test_late_start_without_course_ok(self):
        racers = synth.std_k_racers()
        racers[5] = synth.k_racer("L0", 6, racer_no="4006", course="", st="L .", rt=".  .")
        p = self.k(racers)
        self.assertEqual(p.issues, [])

    def test_empty_bet_line(self):
        pays = synth.std_pays()
        pays[1] = "        複勝     "
        p = self.k(synth.std_k_racers(), pays)
        self.assertEqual(p.issues, [])
        self.assertIn(("place", "none"), [(x["bet_type"], x["payout_kind"]) for x in p.payouts])

    def test_conditions_missing_warns(self):
        p = self.k(synth.std_k_racers(), cond="不明")
        self.assertEqual([(i["check"], i["level"]) for i in p.issues], [("k_conditions", "warn")])


class TestBuild(unittest.TestCase):
    def day(self, b_races, k_races, summary=None):
        b = parse_b.parse_b(synth.b_file(races=b_races), D)
        k = parse_k.parse_k(synth.k_file(races=k_races, summary=summary), D)
        return build.build_day(D, b, k, "bsha", "ksha")

    def kr(self, no, racers=None, pays=None):
        return (no, racers or synth.std_k_racers(), pays if pays is not None else synth.std_pays(), COND, "逃げ")

    def test_ok_day(self):
        t = self.day([(1, synth.std_b_racers())], [self.kr(1)], summary=[(1, "1-2-3", 600)])
        self.assertEqual(t.day_verdict, "ok")
        self.assertEqual(t.races[0]["race_status"], "completed")
        self.assertEqual(t.counts["k_results"], 6)

    def test_result_missing_not_cancelled(self):
        t = self.day([(1, synth.std_b_racers()), (2, synth.std_b_racers())], [self.kr(1)])
        st = {r["race_no"]: r["race_status"] for r in t.races}
        self.assertEqual(st, {1: "completed", 2: "result_missing"})  # K に無いだけで中止にしない
        self.assertEqual(t.day_verdict, "ok")

    def test_k_race_not_in_b_stops(self):
        t = self.day([(1, synth.std_b_racers())], [self.kr(1), self.kr(2)])
        self.assertEqual(levels(t.checks, "k_race_not_in_b"), ["stop"])
        self.assertEqual(t.day_verdict, "stop")

    def test_bk_mismatch_stops(self):
        racers = synth.std_k_racers()
        racers[0] = synth.k_racer("01", 1, racer_no="9999", course="1")
        t = self.day([(1, synth.std_b_racers())], [self.kr(1, racers)])
        self.assertEqual(levels(t.checks, "bk_entry_mismatch"), ["stop"])

    def test_not6_quarantined(self):
        t = self.day([(1, synth.std_b_racers()[:5])], [self.kr(1)])
        self.assertIn("quarantined", levels(t.checks, "b_race_not6"))
        self.assertTrue(t.races[0]["quarantined"])
        self.assertEqual(t.day_verdict, "quarantined")

    def test_trifecta_mismatch_quarantined(self):
        pays = synth.std_pays(order=(2, 1, 3))
        t = self.day([(1, synth.std_b_racers())], [self.kr(1, pays=pays)])
        self.assertEqual(levels(t.checks, "payout_trifecta_mismatch"), ["quarantined"])

    def test_tie_at_fourth_still_checked(self):
        racers = synth.std_k_racers(codes={5: "04"})
        t = self.day([(1, synth.std_b_racers())], [self.kr(1, racers)])
        self.assertEqual(levels(t.checks, "payout_check_skipped"), [])
        self.assertEqual(t.day_verdict, "ok")

    def test_duplicate_key_stops(self):
        t = self.day([(1, synth.std_b_racers()), (1, synth.std_b_racers())], [self.kr(1)])
        self.assertIn("stop", levels(t.checks, "b_race_duplicate_key"))

    def test_cancelled_race_from_k_text(self):
        b = parse_b.parse_b(synth.b_file(races=[(1, synth.std_b_racers()), (2, synth.std_b_racers())]), D)
        k = parse_k.parse_k(synth.k_file(races=[self.kr(1)], tail=["   2R  中 止"]), D)
        t = build.build_day(D, b, k)
        st = {r["race_no"]: (r["race_status"], r["status_evidence"]) for r in t.races}
        self.assertEqual(st[2], ("cancelled", "K: 2R  中 止"))
        self.assertEqual(t.day_verdict, "ok")

    def test_void_race_and_void_payouts(self):
        racers = synth.std_k_racers(codes={3: "S0", 4: "S0", 5: "S0", 6: "00"})
        pays = ["        単勝     1          110  ", "        拡連複   不成立", "        ３連複   不成立"]
        raw = synth.k_file(races=[(1, racers + ["レース不成立".encode("cp932")], pays, COND, "")])
        k = parse_k.parse_k(raw, D)
        self.assertTrue(k.races[0]["race_void"])
        self.assertEqual([(p["bet_type"], p["payout_kind"], p["payout_yen"]) for p in k.payouts][1:],
                         [("wide", "不成立", None), ("trio", "不成立", None)])
        b = parse_b.parse_b(synth.b_file(races=[(1, synth.std_b_racers())]), D)
        t = build.build_day(D, b, k)
        self.assertEqual(t.races[0]["race_status"], "void")
        self.assertEqual(levels(t.checks, "k_payout_line"), [])
        self.assertEqual(levels(t.checks, "k_technique_missing"), [])
        self.assertEqual(levels(t.checks, "k_unknown_code"), ["quarantined"])  # 00 はそのまま隔離

    def test_missing_b_source(self):
        k = parse_k.parse_k(synth.k_file(races=[self.kr(1)]), D)
        t = build.build_day(D, None, k)
        self.assertEqual(t.day_verdict, "stop")
        self.assertEqual(t.races[0]["race_status"], "completed_no_program")


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.t += s


class TestFetch(unittest.TestCase):
    def test_interval_retry_404_and_no_overwrite(self):
        body = b"hello"
        lzh_ok = TestArchive()._fake_lzh(archive.crc16(body), name=b"B230101.TXT")
        calls = []
        script = [ConnectionResetError("x"), (200, lzh_ok), urllib.error.HTTPError("u", 404, "nf", {}, None),
                  (200, lzh_ok + b"X")]

        def opener(url, timeout):
            calls.append(url)
            r = script.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        fc = FakeClock()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(archive.lzh_extract, "extract_all", return_value={"B230101.TXT": body}):
            f = fetch.Fetcher(Path(tmp), opener=opener, interval=3.0, clock=fc.clock, sleep=fc.sleep)
            rec, data = f.get("B", D)                       # 1回目は通信の失敗 → 再試行で成功
            self.assertEqual((rec["capture_status"], rec["crc_ok"], len(calls)), ("saved", True, 2))
            self.assertTrue(f.path_for("B", D).exists())
            rec2, _ = f.get("K", D)                         # 404
            self.assertEqual(rec2["capture_status"], "not_found")
            with self.assertRaises(fetch.RawConflict):      # 中身の違う原本は上書きしない
                f.get("B", D)
            self.assertEqual(f.path_for("B", D).read_bytes(), lzh_ok)
        self.assertTrue(all(s <= 3.0 for s in fc.sleeps) and len(fc.sleeps) == 3)  # 毎回3秒あける
        self.assertEqual(calls[0], "http://www1.mbrace.or.jp/od2/B/202301/b230101.lzh")


class TestBackfill(unittest.TestCase):
    def test_skips_done_days_and_retries_failed(self):
        import io
        from contextlib import redirect_stdout
        from asahi_ai import cli, store
        body_b = synth.b_file(races=[(1, synth.std_b_racers())])
        body_k = synth.k_file(races=[(1, synth.std_k_racers(), synth.std_pays(), COND, "逃げ")])
        calls = []

        def opener(url, timeout):
            calls.append(url)
            if "230102" in url:
                raise urllib.error.HTTPError(url, 404, "nf", {}, None)
            if "230103" in url and len([c for c in calls if "230103" in c]) <= 2:
                raise ConnectionResetError("x")
            return 200, b"LZH:" + url.encode()

        def fake_extract(data):
            name = data.decode().rsplit("/", 1)[1]
            return archive.Member(name, "-lh5-", body_b if "/b2" in data.decode() else
                                  body_k.replace(b"2023/ 1/ 1", f"2023/ 1/{int(name[5:7]):>2}".encode()), 0, 0)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(archive, "extract_single", side_effect=fake_extract), \
                mock.patch.object(fetch, "_default_opener", opener):
            args = ["backfill", "--start", "2023-01-01", "--end", "2023-01-03", "--data", tmp, "--interval", "0"]
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main(args)
            self.assertTrue(store.checks_path(tmp, "2023-01-01").exists())
            self.assertTrue(store.checks_path(tmp, "2023-01-02").exists())   # 開催なし（404）は済み扱い
            self.assertFalse(store.checks_path(tmp, "2023-01-03").exists())  # 取得失敗は次回に取り直す
            self.assertIn('"remaining": 1', out.getvalue())
            calls.clear()
            with redirect_stdout(io.StringIO()):
                cli.main(args)
            self.assertTrue(all("230103" in c for c in calls))              # 済んだ日は取りに行かない


if __name__ == "__main__":
    unittest.main()
