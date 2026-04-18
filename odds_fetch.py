"""
odds_fetch.py  ── ボートレース公式サイトから3連単オッズを取得

デバッグ方法:
  python odds_fetch.py --rno 1 --jcd 04 --hd 20260418 --debug
  → HTMLの生の構造と取得結果を標準出力に表示する
"""

from __future__ import annotations
import time
import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
    "Referer": "https://www.boatrace.jp/",
}
TIMEOUT = 10
RETRY   = 2


# ════════════════════════════════════════════════════════════
# メイン取得関数
# ════════════════════════════════════════════════════════════

def fetch_odds(
    race_no:    int,
    venue_code: str,
    race_date:  str,
    debug:      bool = False,
) -> dict[str, float]:
    """
    3連単オッズを {"1-2-3": 47.2, ...} 形式で返す。
    失敗時は {} を返す。
    """
    params = {
        "rno": str(race_no),
        "jcd": str(venue_code).zfill(2),
        "hd":  race_date,
    }

    res = None
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

    soup = BeautifulSoup(res.content, "html.parser")

    if debug:
        _debug_html(soup)

    odds_map = _parse_v3(soup)

    if odds_map:
        print(f"[odds_fetch] {len(odds_map)}点取得 "
              f"({venue_code} {race_no}R {race_date})")
    else:
        print("[odds_fetch] オッズ取得できず → 暫定推定で代替")

    return odds_map


# ════════════════════════════════════════════════════════════
# パーサー v3: 全手法を試す
# ════════════════════════════════════════════════════════════

def _parse_v3(soup: BeautifulSoup) -> dict[str, float]:
    """
    複数の解析手法を順番に試し、120点最も近い結果を採用する。
    """
    results = []

    # 手法A: テーブル分割 + rowspan 解析
    try:
        r = _method_a_rowspan(soup)
        results.append(("A_rowspan", r))
    except Exception as e:
        print(f"[odds_fetch] 手法A失敗: {e}")

    # 手法B: oddsPoint を順番に取り出して辞書順割り当て
    try:
        r = _method_b_sequential(soup)
        results.append(("B_sequential", r))
    except Exception as e:
        print(f"[odds_fetch] 手法B失敗: {e}")

    # 手法C: 全テキストから数値パターンで抽出
    try:
        r = _method_c_text(soup)
        results.append(("C_text", r))
    except Exception as e:
        print(f"[odds_fetch] 手法C失敗: {e}")

    if not results:
        return {}

    # 120点に最も近い結果を採用
    best_name, best = max(results, key=lambda x: len(x[1]))
    print(f"[odds_fetch] 採用手法: {best_name} ({len(best)}点)")
    return best


# ════════════════════════════════════════════════════════════
# 手法A: rowspan を考慮したテーブル解析
# ════════════════════════════════════════════════════════════

def _method_a_rowspan(soup: BeautifulSoup) -> dict[str, float]:
    """
    3連単テーブルの構造:
      - 6テーブル（1着艇ごと）
      - 各テーブル内で 1着艇番が rowspan=20 の td として最初の列にある
      - 2着艇番が rowspan=4 の td として2列目にある
      - 3着艇番が個別の td として3列目にある
      - oddsPoint が4列目にある

    実際のHTML例:
      <tr>
        <td rowspan="20" class="is-boatColor1">1</td>  ← 1着（1テーブルに1回）
        <td rowspan="4"  class="is-boatColor2">2</td>  ← 2着（5回出現）
        <td class="is-boatColor3">3</td>               ← 3着
        <td class="oddsPoint">47.2</td>
      </tr>
    """
    odds_map: dict[str, float] = {}

    for table in soup.find_all("table"):
        rows = table.select("tbody tr")
        if not rows:
            continue

        # このテーブルに oddsPoint があるか確認
        if not table.find("td", class_="oddsPoint"):
            continue

        first_boat  = None
        second_boat = None

        for tr in rows:
            tds = tr.find_all("td")
            if not tds:
                continue

            # rowspan=20 → 1着艇番
            for td in tds:
                rs = td.get("rowspan", "")
                try:
                    if int(rs) >= 18:   # 20前後を許容
                        v = int(td.get_text(strip=True))
                        if 1 <= v <= 6:
                            first_boat = v
                except (ValueError, TypeError):
                    pass

            # rowspan=4 → 2着艇番
            for td in tds:
                rs = td.get("rowspan", "")
                try:
                    if int(rs) in (3, 4, 5):   # 4前後を許容
                        v = int(td.get_text(strip=True))
                        if 1 <= v <= 6:
                            second_boat = v
                except (ValueError, TypeError):
                    pass

            # oddsPoint → オッズ値
            odds_td = tr.find("td", class_="oddsPoint")
            if not odds_td:
                continue

            # rowspan なし の artColor セル → 3着艇番
            third_boat = None
            for td in tds:
                classes = " ".join(td.get("class", []))
                rs = td.get("rowspan")
                if "boatColor" in classes and rs is None:
                    try:
                        v = int(td.get_text(strip=True))
                        if 1 <= v <= 6:
                            third_boat = v
                    except ValueError:
                        pass

            if first_boat and second_boat and third_boat:
                try:
                    odds_val = float(
                        odds_td.get_text(strip=True).replace(",", "")
                    )
                    combo = f"{first_boat}-{second_boat}-{third_boat}"
                    odds_map[combo] = odds_val
                except ValueError:
                    pass

    return odds_map


# ════════════════════════════════════════════════════════════
# 手法B: oddsPoint 順番割り当て（シンプル・高信頼）
# ════════════════════════════════════════════════════════════

def _method_b_sequential(soup: BeautifulSoup) -> dict[str, float]:
    """
    全 oddsPoint を出現順に取り出し、3連単の辞書順に割り当てる。
    公式サイトは必ず辞書順（1-2-3, 1-2-4, ... 6-5-4）で並んでいる。
    """
    all_odds: list[float] = []
    for td in soup.select("td.oddsPoint"):
        txt = td.get_text(strip=True).replace(",", "")
        try:
            all_odds.append(float(txt))
        except ValueError:
            continue

    if len(all_odds) < 60:
        return {}

    combos = [
        f"{i}-{j}-{k}"
        for i in range(1, 7)
        for j in range(1, 7)
        for k in range(1, 7)
        if i != j and j != k and i != k
    ]

    return {combo: odds for combo, odds in zip(combos, all_odds[:120])}


# ════════════════════════════════════════════════════════════
# 手法C: テキスト全体から「N-N-N タブ 数値」パターンで抽出
# ════════════════════════════════════════════════════════════

def _method_c_text(soup: BeautifulSoup) -> dict[str, float]:
    """
    ページ全体のテキストから「艇番-艇番-艇番」と数値のペアを正規表現で抽出。
    """
    text     = soup.get_text(separator="\t")
    pattern  = re.compile(r"([1-6])-([1-6])-([1-6])\s+([0-9,]+\.?[0-9]*)")
    odds_map : dict[str, float] = {}

    for m in pattern.finditer(text):
        i, j, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if i == j or j == k or i == k:
            continue
        try:
            odds_map[f"{i}-{j}-{k}"] = float(m.group(4).replace(",", ""))
        except ValueError:
            continue

    return odds_map


# ════════════════════════════════════════════════════════════
# デバッグ出力
# ════════════════════════════════════════════════════════════

def _debug_html(soup: BeautifulSoup) -> None:
    """HTML構造の要点を標準出力に表示する"""
    print("\n" + "="*60)
    print("[DEBUG] テーブル一覧")
    tables = soup.find_all("table")
    print(f"  テーブル数: {len(tables)}")

    for ti, table in enumerate(tables):
        rows = table.select("tbody tr")
        op   = table.find_all("td", class_="oddsPoint")
        print(f"\n  Table[{ti}]: {len(rows)}行 / oddsPoint={len(op)}個")
        if rows:
            # 最初の3行を表示
            for ri, tr in enumerate(rows[:3]):
                tds = tr.find_all("td")
                td_info = []
                for td in tds:
                    cls = " ".join(td.get("class", []))
                    rs  = td.get("rowspan", "-")
                    txt = td.get_text(strip=True)[:10]
                    td_info.append(f"[cls={cls} rs={rs} txt={txt}]")
                print(f"    tr[{ri}]: {' | '.join(td_info)}")

    print("\n[DEBUG] oddsPoint 先頭20件")
    for i, td in enumerate(soup.select("td.oddsPoint")[:20]):
        print(f"  [{i:02d}] {td.get_text(strip=True)}")
    print("="*60 + "\n")


# ════════════════════════════════════════════════════════════
# CLI（デバッグ用単体実行）
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--rno",   required=True, help="レース番号")
    parser.add_argument("--jcd",   required=True, help="場コード2桁")
    parser.add_argument("--hd",    required=True, help="日付 YYYYMMDD")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    result = fetch_odds(int(args.rno), args.jcd, args.hd, debug=args.debug)
    print(f"\n取得結果: {len(result)}点")
    # 先頭10件を表示
    for k, v in list(result.items())[:10]:
        print(f"  {k}: {v}")