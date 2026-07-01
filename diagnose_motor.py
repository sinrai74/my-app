#!/usr/bin/env python3
"""
diagnose_motor.py  ── motor_history.csv が生成されない原因を診断する

Usage:
    python diagnose_motor.py
    python diagnose_motor.py --date 20260630   # 特定日を指定
"""
import json, sys, os, requests
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
RESULTS_URL  = "https://boatraceopenapi.github.io/results/v2"
PROGRAMS_URL = "https://boatraceopenapi.github.io/programs/v2"
HEADERS = {"User-Agent": "Mozilla/5.0"}

date = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--date" else \
       (datetime.now(JST) - timedelta(days=1)).strftime("%Y%m%d")

print(f"=== 診断対象日: {date} ===\n")

# ① 結果API
print("① 結果API取得中...")
url = f"{RESULTS_URL}/{date[:4]}/{date}.json"
print(f"   URL: {url}")
try:
    r = requests.get(url, headers=HEADERS, timeout=15)
    print(f"   ステータス: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        results = data.get("results", [])
        print(f"   レース数: {len(results)}")
        if results:
            sample = results[0]
            boats = sample.get("boats", [])
            print(f"   boats型: {type(boats).__name__}  要素数: {len(boats) if isinstance(boats, list) else len(boats) if isinstance(boats, dict) else '?'}")
            print(f"   サンプルboat[0]キー: {list(boats[0].keys()) if isinstance(boats, list) and boats else '取得失敗'}")
            # racer_place_number があるか
            has_place = any("racer_place_number" in b for b in (boats if isinstance(boats, list) else boats.values()))
            has_motor = any("racer_assigned_motor_number" in b or "racer_motor_number" in b for b in (boats if isinstance(boats, list) else boats.values()))
            print(f"   racer_place_number: {'あり ✅' if has_place else 'なし ❌'}")
            print(f"   motor_number系フィールド: {'あり ✅' if has_motor else 'なし ❌'}")
            # モーター番号のフィールド名を特定
            if isinstance(boats, list) and boats:
                motor_keys = [k for k in boats[0].keys() if "motor" in k.lower()]
                print(f"   motorを含むフィールド名: {motor_keys}")
        else:
            print("   ⚠️ results が空です → この日はレース結果がありません")
    else:
        print(f"   ❌ HTTPエラー → results API からデータを取得できません")
        print("   → motor_history.csv は生成されません（正常: レースがない日）")
except Exception as e:
    print(f"   ❌ 例外: {e}")

print()

# ② 出走表API
print("② 出走表API取得中...")
url2 = f"{PROGRAMS_URL}/{date[:4]}/{date}.json"
print(f"   URL: {url2}")
try:
    r2 = requests.get(url2, headers=HEADERS, timeout=15)
    print(f"   ステータス: {r2.status_code}")
    if r2.status_code == 200:
        data2 = r2.json()
        programs = data2.get("programs", data2.get("races", []))
        print(f"   出走表レース数: {len(programs)}")
        if programs:
            boats2 = programs[0].get("boats", [])
            if boats2:
                motor_keys2 = [k for k in (boats2[0].keys() if isinstance(boats2, list) else list(boats2.values())[0].keys()) if "motor" in k.lower()]
                print(f"   motorを含むフィールド名(出走表): {motor_keys2}")
    else:
        print(f"   ❌ HTTPエラー")
except Exception as e:
    print(f"   ❌ 例外: {e}")

print()

# ③ motor_history.csv の現状
print("③ motor_history.csv の現状...")
if os.path.exists("motor_history.csv"):
    import csv
    with open("motor_history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dates = sorted(set(r.get("date","") for r in rows))
    print(f"   行数: {len(rows)}")
    print(f"   日付範囲: {dates[0] if dates else 'なし'} 〜 {dates[-1] if dates else 'なし'}")
else:
    print("   ❌ ファイルが存在しません")
    print("   → x_ranking.py --update-history が一度も成功していないか、")
    print("      成功しても git push されていない可能性があります")

print()

# ④ notify_arashi.py のインポート確認
print("④ notify_arashi.py のインポート確認...")
try:
    import notify_arashi
    print("   ✅ インポート成功")
except Exception as e:
    print(f"   ❌ インポート失敗: {e}")
    print("   → x_ranking.py も失敗しています（from notify_arashi import ... が通らない）")

print()
print("=== 診断完了 ===")
