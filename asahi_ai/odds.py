"""3連単オッズの取得（公式 boatrace.jp の odds3t ページ）。予測には使わない（期待値・購入判断の研究用）。

解析は my-app の odds_fetch.py のラベル方式（2026-10 の取り違えの不具合の修正後の方式）と同じ考え方：
表の見出し（1着）と各セル（2着・3着）の艇番から組番を読む。位置だけで組番を決めない。
構造が想定と違えば例外にして、空の結果として記録する（推定で埋めない）。
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request

BASE = "https://www.boatrace.jp/owpc/pc/race/odds3t"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; asahi-ai-odds/1)", "Accept-Language": "ja,en;q=0.9",
           "Referer": "https://www.boatrace.jp/"}


class OddsTableError(ValueError):
    pass


def url_for(rno: int, jcd: str, hd: str) -> str:
    return f"{BASE}?rno={int(rno)}&jcd={str(jcd).zfill(2)}&hd={hd}"


def fetch_html(url: str, timeout: float = 20) -> tuple[int | None, bytes | None, str | None]:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), None
    except urllib.error.HTTPError as e:
        return e.code, None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


def _odds_value(td) -> float | None:
    txt = td.get_text(strip=True).replace(",", "")
    try:
        v = float(txt)
    except ValueError:
        return None
    return v if v > 0 else None


def _boat_no(td) -> int:
    txt = td.get_text(strip=True)
    if not txt.isdigit() or not 1 <= int(txt) <= 6:
        raise OddsTableError(f"艇番のセルが不正: {txt!r}")
    return int(txt)


def parse_trifecta(html: bytes | str) -> dict[str, float]:
    """3連単の表 → {"1-2-3": 47.2, ...}。数値でないセル（欠場・発売前など）の組番は含めない。
    表が無い・構造が違うときは OddsTableError。"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    tables = [t for t in soup.find_all("table") if t.find("td", class_="oddsPoint")]
    if len(tables) != 1:
        raise OddsTableError(f"oddsPoint を含む表が {len(tables)} 個ある")
    table = tables[0]
    firsts = [int(th.get_text(strip=True)) for th in table.select("thead th") if th.get_text(strip=True).isdigit()]
    if sorted(firsts) != [1, 2, 3, 4, 5, 6]:
        raise OddsTableError(f"thead の1着艇番が不正: {firsts}")
    n = len(firsts)
    second: list[int | None] = [None] * n
    left = [0] * n
    seen: set[str] = set()
    out: dict[str, float] = {}
    rows = table.select("tbody tr")
    if not rows:
        raise OddsTableError("tbody に行がない")
    for tr in rows:
        tds = tr.find_all("td")
        i = 0
        for col in range(n):
            if left[col] == 0:
                if i >= len(tds):
                    raise OddsTableError("2着のセルが足りない")
                cell = tds[i]
                try:
                    span = int(cell.get("rowspan", "1"))
                except ValueError as e:
                    raise OddsTableError("rowspan が数値でない") from e
                second[col] = _boat_no(cell)
                left[col] = span
                i += 1
            if i + 1 >= len(tds):
                raise OddsTableError("3着・オッズのセルが足りない")
            third_td, odds_td = tds[i], tds[i + 1]
            if "oddsPoint" not in (odds_td.get("class") or []):
                raise OddsTableError("オッズのセルの位置が想定と違う")
            a, b, c = firsts[col], second[col], _boat_no(third_td)
            if len({a, b, c}) != 3:
                raise OddsTableError(f"組番が不正: {a}-{b}-{c}")
            combo = f"{a}-{b}-{c}"
            if combo in seen:
                raise OddsTableError(f"組番が重複: {combo}")
            seen.add(combo)
            v = _odds_value(odds_td)
            if v is not None:
                out[combo] = v
            i += 2
            left[col] -= 1
        if i != len(tds):
            raise OddsTableError("行のセル数が想定と違う")
    if len(seen) != 120:
        raise OddsTableError(f"組番が120通りでない: {len(seen)}")
    return out


def page_update_text(html: bytes | str) -> str | None:
    """ページに表示されるオッズの更新時刻（例「オッズ更新時間 14:52」「締切時オッズ」）の文字。無ければ None。"""
    s = html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    m = re.search(r"(締切時オッズ|オッズ更新時間[^0-9]{0,10}\d{1,2}:\d{2}|\d{1,2}:\d{2}\s*現在)", s)
    return m.group(1) if m else None
