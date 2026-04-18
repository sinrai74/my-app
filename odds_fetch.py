
"""
odds_fetch.py  ── ボートレース公式サイトから3連単オッズを取得

取得URL:
  https://www.boatrace.jp/owpc/pc/race/odds3t?rno={R}&jcd={場コード2桁}&hd={YYYYMMDD}

注意:
  - 締切30分前〜直前が最終オッズに最も近い
  - 過剰アクセスは厳禁（1回の予想で1リクエストのみ）
  - サイトのHTML構造が変わった場合は CSSセレクタを修正する
"""

from __future__ import annotations
import time
import requests
from bs4 import BeautifulSoup


# ── 定数 ─────────────────────────────────────────────────────
BASE_URL   = "https://www.boatrace.jp/owpc/pc/race/odds3t"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
TIMEOUT    = 8   # 秒
RETRY      = 2   # リトライ回数


def fetch_odds(
    race_no:    int,
    venue_code: str,
    race_date:  str,
) -> dict[str, float]:
    """
    3連単オッズを取得して {組番: オッズ} の辞書で返す。

    Args:
        race_no:    レース番号 (1〜12)
        venue_code: 場コード2桁文字列 (例: "04" = 平和島)
        race_date:  日付文字列 YYYYMMDD (例: "20260418")

    Returns:
        {"1-2-3": 47.2, "1-2-4": 60.3, ...}  ※取得失敗時は {}
    """
    url    = BASE_URL
    params = {
        "rno": str(race_no),
        "jcd": str(venue_code).zfill(2),
        "hd":  race_date,
    }

    for attempt in range(RETRY + 1):
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            res.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < RETRY:
                time.sleep(1.5)
                continue
            print(f"[odds_fetch] 取得失敗: {e}")
            return {}

    soup = BeautifulSoup(res.content, "html.parser")
    return _parse_odds3t(soup)


def _parse_odds3t(soup: BeautifulSoup) -> dict[str, float]:
    """
    3連単オッズページのHTMLを解析して辞書に変換。

    HTML構造（2024年以降の公式サイト）:
      各行 <tr> に対して
        - 1着艇番: <td class="is-boatColor{N}"> の数字
        - オッズ値: <td class="oddsPoint"> のテキスト
      1着ごとに20行（2着×3着 の組み合わせ: 5×4=20通り）

    フォールバック:
      oddsPoint が取れなかった場合、数値テキストを持つtdから推定
    """
    odds_map: dict[str, float] = {}

    # ── 方法①: oddsPoint クラスから取得（メイン） ──────────
    try:
        rows = soup.select("tbody tr")
        if rows:
            first_boat = 1
            count_in_block = 0  # 1着艇番ブロック内のカウント (0〜19)

            # 1着ごとのブロックを特定するため、艇番カラーセルを検索
            for row in rows:
                # 艇番カラーセル（is-boatColor1〜6）があれば1着艇番更新
                for c in range(1, 7):
                    boat_cell = row.select_one(f"td.is-boatColor{c}")
                    if boat_cell and boat_cell.get_text(strip=True).isdigit():
                        first_boat = int(boat_cell.get_text(strip=True))
                        break

                # 2着・3着の番号セル
                boat_cells = row.select("td[class*='is-boatColor']")
                nums = [
                    int(td.get_text(strip=True))
                    for td in boat_cells
                    if td.get_text(strip=True).isdigit()
                ]

                # oddsPoint セル
                odds_cells = row.select("td.oddsPoint")
                odds_vals  = []
                for td in odds_cells:
                    txt = td.get_text(strip=True).replace(",", "")
                    try:
                        odds_vals.append(float(txt))
                    except ValueError:
                        odds_vals.append(0.0)

                # 2着・3着の組が揃っている行だけ登録
                if len(nums) >= 2 and odds_vals:
                    second, third = nums[0], nums[1]
                    combo = f"{first_boat}-{second}-{third}"
                    odds_map[combo] = odds_vals[0]

    except Exception as e:
        print(f"[odds_fetch] パース方法①失敗: {e}")

    # ── 方法②: テーブル全体から行列で取得（フォールバック） ──
    if len(odds_map) < 60:
        try:
            odds_map = _parse_fallback(soup)
        except Exception as e:
            print(f"[odds_fetch] パース方法②失敗: {e}")

    if odds_map:
        print(f"[odds_fetch] {len(odds_map)}点取得")
    else:
        print("[odds_fetch] オッズ取得できず（フロント側オッズで代替）")

    return odds_map


def _parse_fallback(soup: BeautifulSoup) -> dict[str, float]:
    """
    フォールバック: テーブルの行列順から3連単を再構築。
    120通り全て揃っていれば行番号から組番を推定できる。
    """
    odds_map: dict[str, float] = {}
    all_odds = []

    for td in soup.select("td.oddsPoint"):
        txt = td.get_text(strip=True).replace(",", "")
        try:
            all_odds.append(float(txt))
        except ValueError:
            continue

    if len(all_odds) < 120:
        return {}

    # 3連単120通りを辞書順で生成
    combos = [
        f"{i}-{j}-{k}"
        for i in range(1, 7)
        for j in range(1, 7)
        for k in range(1, 7)
        if i != j and j != k and i != k
    ]

    for combo, odds_val in zip(combos, all_odds[:120]):
        odds_map[combo] = odds_val

    return odds_map