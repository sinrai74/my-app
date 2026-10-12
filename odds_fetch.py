"""
odds_fetch.py  ── ボートレース公式サイトから3連単オッズを取得

デバッグ方法:
  python odds_fetch.py --rno 1 --jcd 04 --hd 20260418 --debug
  → HTMLの生の構造と取得結果を標準出力に表示する
"""

from __future__ import annotations
import logging
import time
import re
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

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
            log.warning("[odds_fetch] 取得失敗(HTTP): %s %sR %s: %s",
                        venue_code, race_no, race_date, e)
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
    表のラベル（1着・2着・3着の艇番）から組番を読む。

    ラベル方式で表の構造を認識できたときは、その結果だけを返す。
    欠場・発売前など数値でないセルの組番はキーを入れない（既存の契約どおり）。
    出現順に辞書順の組番を割り当てる手法Bは、ページの並びと一致しないため
    採用しない（docs/incidents/2026-10-09_odds-parser-combo-mismatch.md）。
    """
    try:
        labeled = _method_labels(soup)
    except OddsTableError as e:
        labeled = None
        log.warning("[odds_fetch] 解析失敗(3連単表の構造を認識できない): %s", e)

    if labeled is not None:
        print(f"[odds_fetch] 採用手法: labels ({len(labeled)}点)")
        if not labeled:
            log.warning("[odds_fetch] 有効なオッズ0件(表は認識。欠場・発売前など)")
        return labeled

    return _parse_label_based_fallback(soup)


def _parse_label_based_fallback(soup: BeautifulSoup) -> dict[str, float]:
    """
    ラベル方式で表を認識できなかった場合の旧来の手法。
    組番をラベル（艇番のセル・テキスト）から読む手法A・Cだけを試す。
    位置だけで組番を決める手法Bは使わない。
    """
    results = []

    # 手法A: テーブル分割 + rowspan 解析
    try:
        r = _method_a_rowspan(soup)
        results.append(("A_rowspan", r))
    except Exception as e:
        print(f"[odds_fetch] 手法A失敗: {e}")

    # 手法C: 全テキストから数値パターンで抽出
    try:
        r = _method_c_text(soup)
        results.append(("C_text", r))
    except Exception as e:
        print(f"[odds_fetch] 手法C失敗: {e}")

    if not results:
        log.warning("[odds_fetch] 解析失敗(代替手法もすべて例外)")
        return {}

    best_name, best = max(results, key=lambda x: len(x[1]))
    print(f"[odds_fetch] 採用手法: {best_name} ({len(best)}点)")
    if best:
        log.warning("[odds_fetch] ラベル方式で表を認識できず代替手法 %s を使用 (%d点)",
                    best_name, len(best))
    else:
        log.warning("[odds_fetch] 解析失敗(オッズを1件も読めない)")
    return best


# ════════════════════════════════════════════════════════════
# ラベル方式: 1つの表の thead(1着) と各セル(2着・3着)から組番を読む
# ════════════════════════════════════════════════════════════

class OddsTableError(ValueError):
    """3連単オッズ表の構造を認識できない"""


def _odds_value(td) -> float | None:
    """オッズのセルを数値にする。欠場・発売前など数値でなければ None"""
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


def _method_labels(soup: BeautifulSoup) -> dict[str, float] | None:
    """
    3連単オッズ表（1つの table）から組番とオッズを読む。

    表の形:
      thead: 1着艇番が6列（th の数字）
      tbody: 1行に1着艇ごとの [2着(rowspanあり)] 3着 オッズ が6組並ぶ。
             2着のセルは rowspan で4行分にまたがる。

    戻り値:
      dict  … 表を認識できた。数値でないセル（欠場等）の組番は含めない
      None  … oddsPoint を含む表がない（このページ形式ではない）
    例外 OddsTableError … 表はあるが構造が想定と違う
    """
    tables = [t for t in soup.find_all("table") if t.find("td", class_="oddsPoint")]
    if not tables:
        return None
    if len(tables) != 1:
        raise OddsTableError(f"oddsPoint を含む表が {len(tables)} 個ある")
    table = tables[0]

    firsts = [int(th.get_text(strip=True))
              for th in table.select("thead th")
              if th.get_text(strip=True).isdigit()]
    if sorted(firsts) != [1, 2, 3, 4, 5, 6]:
        raise OddsTableError(f"thead の1着艇番が不正: {firsts}")

    n_cols = len(firsts)
    second: list[int | None] = [None] * n_cols
    second_left = [0] * n_cols
    seen: set[str] = set()
    odds_map: dict[str, float] = {}

    rows = table.select("tbody tr")
    if not rows:
        raise OddsTableError("tbody に行がない")

    for tr in rows:
        tds = tr.find_all("td")
        i = 0
        for col in range(n_cols):
            if second_left[col] == 0:
                if i >= len(tds):
                    raise OddsTableError("2着のセルが足りない")
                cell = tds[i]
                try:
                    span = int(cell.get("rowspan", "1"))
                except ValueError:
                    raise OddsTableError("rowspan が数値でない")
                second[col] = _boat_no(cell)
                second_left[col] = span
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
                odds_map[combo] = v
            i += 2
            second_left[col] -= 1
        if i != len(tds):
            raise OddsTableError("行のセル数が想定と違う")

    if len(seen) != 120:
        raise OddsTableError(f"組番が120通りでない: {len(seen)}")
    return odds_map


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