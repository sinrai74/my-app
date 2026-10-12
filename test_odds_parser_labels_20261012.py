"""
odds_fetch の3連単オッズ解析（ラベル方式）のテスト。

docs/incidents/2026-10-09_odds-parser-combo-mismatch.md と
docs/incidents/2026-10-12_odds-parser-combo-mismatch-supplement.md の修正の確認用。
公式ページの HTML は使わず、表の構造を再現した合成 HTML で確認する。
各セルの値に組番を埋め込み（例: 2-1-3 → 213.5）、取り違えを検出できるようにしている。

実行: python -m unittest test_odds_parser_labels_20261012 -v
"""
from __future__ import annotations

import re
import unittest
from unittest import mock

import requests
from bs4 import BeautifulSoup

import odds_fetch


def _pairs(first: int, order: str) -> list[tuple[int, int]]:
    pairs = [(b, c) for b in range(1, 7) for c in range(1, 7) if len({first, b, c}) == 3]
    if order == "second_asc_third_desc":
        pairs.sort(key=lambda x: (x[0], -x[1]))
    return pairs


def _value(a: int, b: int, c: int) -> str:
    return f"{a}{b}{c}.5"


def _page(firsts=(1, 2, 3, 4, 5, 6), order="lex", absent=(), override=None,
          with_thead=True, duplicate_first_row=False) -> BeautifulSoup:
    """実ページと同じ形の3連単表（1つの table、thead に1着、各列20行）を作る"""
    override = override or {}
    html = ['<table class="dummy"><tbody><tr><td>1R</td><td>10:47</td></tr></tbody></table>']
    html.append("<table>")
    if with_thead:
        html.append("<thead><tr>")
        for a in firsts:
            html.append(f'<th class="is-boatColor{a}">{a}</th>')
            html.append(f'<th class="is-boatColor{a}" colspan="2">選手{a}</th>')
        html.append("</tr></thead>")
    html.append("<tbody>")
    for r in range(20):
        html.append("<tr>")
        for a in firsts:
            b, c = _pairs(a, order)[r]
            if duplicate_first_row and r == 1 and a == firsts[0]:
                b, c = _pairs(a, order)[0]
            if r % 4 == 0:
                html.append(f'<td class="is-boatColor{b}" rowspan="4">{b}</td>')
            html.append(f'<td class="is-boatColor{c}">{c}</td>')
            if {a, b, c} & set(absent):
                html.append('<td class="oddsPoint is-miss is-fColor1">欠場</td>')
            else:
                v = override.get((a, b, c), _value(a, b, c))
                html.append(f'<td class="oddsPoint ">{v}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")
    return BeautifulSoup("".join(html), "html.parser")


def _expected(absent=()) -> dict[str, float]:
    return {
        f"{a}-{b}-{c}": float(_value(a, b, c))
        for a in range(1, 7) for b in range(1, 7) for c in range(1, 7)
        if len({a, b, c}) == 3 and not ({a, b, c} & set(absent))
    }


def _parse(soup: BeautifulSoup) -> dict[str, float]:
    with mock.patch("builtins.print"):
        return odds_fetch._parse_v3(soup)


class TestLabelParsing(unittest.TestCase):

    def test_normal_120_all_match(self):
        self.assertEqual(_parse(_page()), _expected())

    def test_order_changes_follow_labels(self):
        # thead の並びが逆、列の中の3着が降順でも、ラベルどおりに読む
        got = _parse(_page(firsts=(6, 5, 4, 3, 2, 1), order="second_asc_third_desc"))
        self.assertEqual(got, _expected())

    def test_absent_boat6_has_no_keys(self):
        got = _parse(_page(absent=(6,)))
        self.assertEqual(got, _expected(absent=(6,)))
        self.assertEqual(len(got), 60)
        self.assertFalse(any("6" in k for k in got))

    def test_absent_boat1_has_no_keys(self):
        got = _parse(_page(absent=(1,)))
        self.assertEqual(got, _expected(absent=(1,)))
        self.assertFalse(any("1" in k for k in got))

    def test_two_absent_boats(self):
        got = _parse(_page(absent=(2, 5)))
        self.assertEqual(got, _expected(absent=(2, 5)))
        self.assertEqual(len(got), 24)

    def test_comma_value_and_non_numeric_cells(self):
        soup = _page(override={(1, 2, 3): "1,234.5", (2, 1, 3): "-", (3, 1, 2): "0.0"})
        got = _parse(soup)
        self.assertEqual(got["1-2-3"], 1234.5)
        self.assertNotIn("2-1-3", got)
        self.assertNotIn("3-1-2", got)
        self.assertEqual(len(got), 118)

    def test_contract_keys_and_types(self):
        got = _parse(_page(absent=(4,)))
        self.assertIsInstance(got, dict)
        for k, v in got.items():
            self.assertRegex(k, r"^[1-6]-[1-6]-[1-6]$")
            self.assertEqual(len(set(k.split("-"))), 3)
            self.assertIsInstance(v, float)
            self.assertGreater(v, 0)


class TestStructureErrors(unittest.TestCase):
    """構造が想定と違うときは、位置で割り当てず空を返し、WARNING を出す"""

    def assert_parse_failure(self, soup):
        with self.assertLogs("odds_fetch", level="WARNING") as cm:
            got = _parse(soup)
        self.assertEqual(got, {})
        self.assertTrue(any("解析失敗" in m for m in cm.output), cm.output)

    def test_duplicate_combo(self):
        self.assert_parse_failure(_page(duplicate_first_row=True))

    def test_missing_thead(self):
        self.assert_parse_failure(_page(with_thead=False))

    def test_invalid_first_boats(self):
        self.assert_parse_failure(_page(firsts=(1, 2, 3, 4, 5, 5)))


class TestFailureLogging(unittest.TestCase):
    """戻り値は従来どおり {}。取得失敗・解析失敗・有効0件を WARNING で区別する"""

    def test_http_failure(self):
        with mock.patch.object(odds_fetch.requests, "get",
                               side_effect=requests.ConnectionError("down")), \
             mock.patch.object(odds_fetch.time, "sleep"), \
             mock.patch("builtins.print"), \
             self.assertLogs("odds_fetch", level="WARNING") as cm:
            got = odds_fetch.fetch_odds(1, "02", "20261009")
        self.assertEqual(got, {})
        self.assertTrue(any("取得失敗" in m for m in cm.output), cm.output)

    def test_all_cells_non_numeric_is_not_parse_failure(self):
        soup = _page(absent=(1, 2, 3, 4, 5, 6))
        with self.assertLogs("odds_fetch", level="WARNING") as cm:
            got = _parse(soup)
        self.assertEqual(got, {})
        self.assertTrue(any("有効なオッズ0件" in m for m in cm.output), cm.output)
        self.assertFalse(any("解析失敗" in m for m in cm.output), cm.output)

    def test_page_without_odds_table(self):
        soup = BeautifulSoup("<html><body><p>データがありません</p></body></html>", "html.parser")
        with self.assertLogs("odds_fetch", level="WARNING") as cm:
            got = _parse(soup)
        self.assertEqual(got, {})
        self.assertTrue(any("解析失敗" in m for m in cm.output), cm.output)


if __name__ == "__main__":
    unittest.main()
