"""
odds_fetch.py  ── ボートレース公式サイトから3連単オッズを取得

【公式サイトの3連単テーブル構造】
  URL: https://www.boatrace.jp/owpc/pc/race/odds3t?rno=R&jcd=XX&hd=YYYYMMDD

  テーブルは1着艇ごとに1つ（合計6テーブル）。
  各テーブルのtbodyに tr が20行（2着×3着 = 5×4 = 20通り）。

  各 tr の構造:
    <tr>
      <td class="is-boatColor{N}">  ← 2着艇番（Nは1〜6）
      <td class="is-boatColor{N}">  ← 3着艇番（Nは1〜6）
      <td class="oddsPoint">47.2</td>  ← オッズ値
    </tr>

  1着艇番は各テーブルの直前の見出し行 or th から取得する。
  実際には table.is-w748 などのクラスが付いた6つのテーブルが並ぶ。

注意:
  - 締切30分前〜直前が最終オッズに最も近い
  - 過剰アクセスは厳禁（1予想につき1リクエストのみ）
"""

from __future__ import annotations
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 10
RETRY   = 2


def fetch_odds(
    race_no:    int,
    venue_code: str,
    race_date:  str,
) -> dict[str, float]:
    """
    3連単オッズを {"1-2-3": 47.2, ...} 形式で返す。
    失敗時は {} を返す。

    Args:
        race_no:    レース番号 1〜12
        venue_code: 場コード2桁 (例: "04")
        race_date:  YYYYMMDD (例: "20260418")
    """
    params = {
        "rno": str(race_no),
        "jcd": str(venue_code).zfill(2),
        "hd":  race_date,
    }

    for attempt in range(RETRY + 1):
        try:
            res = requests.get(
                BASE_URL, params=params,
                headers=HEADERS, timeout=TIMEOUT
            )
            res.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < RETRY:
                time.sleep(2.0)
                continue
            print(f"[odds_fetch] HTTP失敗: {e}")
            return {}

    soup     = BeautifulSoup(res.content, "html.parser")
    odds_map = _parse(soup)

    if odds_map:
        print(f"[odds_fetch] {len(odds_map)}点取得 "
              f"({venue_code} {race_no}R {race_date})")
    else:
        print("[odds_fetch] オッズ取得できず → 暫定推定で代替")

    return odds_map


def _parse(soup: BeautifulSoup) -> dict[str, float]:
    """
    公式サイト3連単ページのHTMLを解析。

    ページ構造:
      6つの <table> が並ぶ（1着=1〜6 それぞれ1テーブル）。
      各テーブルの <caption> や直前の <h3> などに1着艇番の記載はなく、
      テーブルの出現順が1着艇番に対応している（1番目=1着1、...6番目=1着6）。
      各 <tr> には2着・3着の艇番セルと oddsPoint セルが含まれる。

      艇番セル: <td class="is-boatColor1">2</td>  ← クラスの数字≠艇番、テキストが艇番
    """
    odds_map: dict[str, float] = {}

    # ── 全テーブルを取得 ────────────────────────────────────
    # oddsPoint を含むテーブルのみ対象にする
    tables = [
        t for t in soup.find_all("table")
        if t.find("td", class_="oddsPoint")
    ]

    if not tables:
        return {}

    # テーブルの出現順 = 1着艇番（1〜6）
    for first_idx, table in enumerate(tables[:6]):
        first_boat = first_idx + 1  # 1着艇番

        for tr in table.select("tbody tr"):
            # 艇番セル: class が "is-boatColor*" のもの
            boat_cells = [
                td for td in tr.find_all("td")
                if td.get("class") and
                any("is-boatColor" in c for c in td.get("class", []))
            ]

            # oddsPoint セル
            odds_cell = tr.find("td", class_="oddsPoint")

            # 艇番2つ・オッズ1つが揃っている行だけ処理
            if len(boat_cells) < 2 or not odds_cell:
                continue

            try:
                second_boat = int(boat_cells[0].get_text(strip=True))
                third_boat  = int(boat_cells[1].get_text(strip=True))
            except ValueError:
                continue

            odds_txt = odds_cell.get_text(strip=True).replace(",", "")
            try:
                odds_val = float(odds_txt)
            except ValueError:
                continue

            # 組番を登録（1着-2着-3着）
            combo = f"{first_boat}-{second_boat}-{third_boat}"
            odds_map[combo] = odds_val

    # ── 取得数チェック（120通りが正常）──────────────────────
    if len(odds_map) < 60:
        print(f"[odds_fetch] 取得数が少ない({len(odds_map)}点) → フォールバック試行")
        fallback = _parse_fallback(soup)
        if len(fallback) > len(odds_map):
            return fallback

    return odds_map


def _parse_fallback(soup: BeautifulSoup) -> dict[str, float]:
    """
    フォールバック: 全 oddsPoint を順番に取り出して
    3連単120通りの辞書順に割り当てる。
    """
    all_odds: list[float] = []
    for td in soup.select("td.oddsPoint"):
        txt = td.get_text(strip=True).replace(",", "")
        try:
            all_odds.append(float(txt))
        except ValueError:
            continue

    if len(all_odds) < 120:
        return {}

    # 3連単120通りを辞書順で生成（公式サイトの表示順と一致）
    combos = [
        f"{i}-{j}-{k}"
        for i in range(1, 7)
        for j in range(1, 7)
        for k in range(1, 7)
        if i != j and j != k and i != k
    ]

    return {combo: odds for combo, odds in zip(combos, all_odds[:120])}