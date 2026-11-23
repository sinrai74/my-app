"""3連単オッズの解析（ラベル方式）と記録の試験。合成 HTML（各セルに組番を埋め込み、取り違えを検出できる形）。"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAS_BS4 = importlib.util.find_spec("bs4") is not None
from asahi_ai import odds, odds_capture  # noqa: E402


def _pairs(first, order):
    pairs = [(b, c) for b in range(1, 7) for c in range(1, 7) if len({first, b, c}) == 3]
    if order == "rev":
        pairs.sort(key=lambda x: (x[0], -x[1]))
    return pairs


def page(firsts=(1, 2, 3, 4, 5, 6), order="lex", absent=(), override=None, with_thead=True, dup=False):
    override = override or {}
    h = ['<html><body><p>オッズ更新時間 14:52</p><table class="dummy"><tbody><tr><td>1R</td></tr></tbody></table><table>']
    if with_thead:
        h.append("<thead><tr>" + "".join(f'<th>{a}</th><th colspan="2">選手{a}</th>' for a in firsts) + "</tr></thead>")
    h.append("<tbody>")
    for r in range(20):
        h.append("<tr>")
        for a in firsts:
            b, c = _pairs(a, order)[r]
            if dup and r == 1 and a == firsts[0]:
                b, c = _pairs(a, order)[0]
            if r % 4 == 0:
                h.append(f'<td rowspan="4">{b}</td>')
            h.append(f"<td>{c}</td>")
            if {a, b, c} & set(absent):
                h.append('<td class="oddsPoint is-miss">欠場</td>')
            else:
                h.append(f'<td class="oddsPoint">{override.get((a, b, c), f"{a}{b}{c}.5")}</td>')
        h.append("</tr>")
    h.append("</tbody></table></body></html>")
    return "".join(h)


def expected(absent=()):
    return {f"{a}-{b}-{c}": float(f"{a}{b}{c}.5") for a in range(1, 7) for b in range(1, 7) for c in range(1, 7)
            if len({a, b, c}) == 3 and not ({a, b, c} & set(absent))}


@unittest.skipUnless(HAS_BS4, "bs4 が無い")
class TestParse(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(odds.parse_trifecta(page()), expected())

    def test_order_follows_labels(self):
        self.assertEqual(odds.parse_trifecta(page(firsts=(6, 5, 4, 3, 2, 1), order="rev")), expected())

    def test_absent(self):
        got = odds.parse_trifecta(page(absent=(2, 5)))
        self.assertEqual(got, expected(absent=(2, 5)))
        self.assertEqual(len(got), 24)

    def test_non_numeric(self):
        got = odds.parse_trifecta(page(override={(1, 2, 3): "1,234.5", (2, 1, 3): "-", (3, 1, 2): "0.0"}))
        self.assertEqual((got["1-2-3"], "2-1-3" in got, "3-1-2" in got, len(got)), (1234.5, False, False, 118))

    def test_structure_errors(self):
        for kw in ({"dup": True}, {"with_thead": False}, {"firsts": (1, 2, 3, 4, 5, 5)}):
            with self.assertRaises(odds.OddsTableError):
                odds.parse_trifecta(page(**kw))
        with self.assertRaises(odds.OddsTableError):
            odds.parse_trifecta("<html>発売前</html>")

    def test_update_text(self):
        self.assertEqual(odds.page_update_text(page()), "オッズ更新時間 14:52")


class TestCapture(unittest.TestCase):
    def _pred(self, tmp, n=30):
        p = Path(tmp) / "predictions" / "2026" / "2026-11-24.jsonl"
        p.parent.mkdir(parents=True)
        rows = []
        for i in range(n):
            pw = [0.9 - i * 0.02] + [(0.1 + i * 0.02) / 5] * 5
            tri = [1 / 120] * 120
            rows.append({"race_id": f"202611240{i % 9 + 1}{i // 9 + 1:02d}", "p_win": pw, "tri": tri,
                         "race_closed_at": f"2026-11-24 {10 + i // 6:02d}:{(i * 7) % 60:02d}:00"})
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    def test_targets_union_top20(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = odds_capture.targets(self._pred(tmp), top_n=5)
            self.assertLessEqual(len(t), 10)
            self.assertTrue(all("danger" in x["flags"] or "upset" in x["flags"] for x in t))
            self.assertEqual([x["closed_at"] for x in t], sorted(x["closed_at"] for x in t))

    @unittest.skipUnless(HAS_BS4, "bs4 が無い")
    def test_run_waits_and_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._pred(tmp, n=6)
            clock = {"t": dt.datetime(2026, 11, 24, 8, 0, tzinfo=odds_capture.JST)}
            def now():
                return clock["t"]
            def sleep(s):
                clock["t"] += dt.timedelta(seconds=s)
            saved = []
            with mock.patch.object(odds, "fetch_html", return_value=(200, page().encode(), None)), \
                    mock.patch.object(odds_capture, "_save_and_push", side_effect=lambda d, p, r: (saved.append(r), p.parent.mkdir(parents=True, exist_ok=True), p.write_text("x"))):
                st = odds_capture.run(Path(tmp), dt.date(2026, 11, 24), 600, sleep=sleep, now=now)
            self.assertEqual(st["targets"], 6)
            self.assertEqual(st["captured"], 12)                     # 6レース × 2回
            self.assertTrue(all(r["n_odds"] == 120 for r in saved))
            self.assertTrue(all(2.9 <= r["minutes_before"] <= 10.1 for r in saved))  # 締切の 10 分前・3 分前
            self.assertEqual({r["snapshot"] for r in saved}, {"m10", "m3"})

    @unittest.skipUnless(HAS_BS4, "bs4 が無い")
    def test_stop_before_long_wait_and_resume(self):
        """区切りの時刻より後の取得は待たずに残し（実行の時間の上限を超えない）、次の実行で続きを取る。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._pred(tmp, n=6)   # 締切 10:00〜10:35 → 取得の予定は 09:50〜10:32
            clock = {"t": dt.datetime(2026, 11, 24, 8, 0, tzinfo=odds_capture.JST)}
            def now():
                return clock["t"]
            def sleep(s):
                clock["t"] += dt.timedelta(seconds=s)
            saved = []
            with mock.patch.object(odds, "fetch_html", return_value=(200, page().encode(), None)), \
                    mock.patch.object(odds_capture, "_save_and_push", side_effect=lambda d, p, r: (saved.append(r), p.parent.mkdir(parents=True, exist_ok=True), p.write_text("x"))):
                st = odds_capture.run(Path(tmp), dt.date(2026, 11, 24), 150, sleep=sleep, now=now)   # 区切り 10:30
                self.assertEqual(st["remaining"], 1)
                self.assertLessEqual(clock["t"], dt.datetime(2026, 11, 24, 10, 31, tzinfo=odds_capture.JST))
                st2 = odds_capture.run(Path(tmp), dt.date(2026, 11, 24), 150, sleep=sleep, now=now)
            self.assertEqual(st["captured"] + st2["captured"], 12)
            self.assertEqual(st2["remaining"], 0)
            self.assertEqual(len({(r["race_id"], r["snapshot"]) for r in saved}), 12)

    @unittest.skipUnless(HAS_BS4, "bs4 が無い")
    def test_long_wait_beyond_limit_is_not_slept(self):
        """朝早くに起動して最初の取得が区切りより後なら、待たずに次の実行へ回す。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._pred(tmp, n=6)
            clock = {"t": dt.datetime(2026, 11, 24, 3, 0, tzinfo=odds_capture.JST)}
            slept = []
            st = odds_capture.run(Path(tmp), dt.date(2026, 11, 24), 330, sleep=slept.append, now=lambda: clock["t"])
            self.assertEqual(st["remaining"], 12)
            self.assertEqual(slept, [])


if __name__ == "__main__":
    unittest.main()
