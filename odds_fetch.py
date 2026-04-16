import requests
from bs4 import BeautifulSoup

def fetch_odds():
    odds_map = {}

    try:
        url = "https://odds.kyotei24.jp/"  # トップ

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        # 3連単オッズのテーブル取得
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cols = row.find_all("td")

                if len(cols) >= 2:
                    combo = cols[0].text.strip()
                    odds_text = cols[1].text.strip()

                    # 数字チェック
                    try:
                        odds = float(odds_text)
                        if "-" in combo:
                            odds_map[combo] = odds
                    except:
                        continue

    except Exception as e:
        print("オッズ取得失敗:", e)
        return {}

    return odds_map