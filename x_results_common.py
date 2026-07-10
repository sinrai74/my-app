#!/usr/bin/env python3
"""
x_results_common.py  ── AI実績ページ 共通集計ロジック

公開用（x_results_public.py）と開発用（x_results_developer.py）の
両方から利用する、データ収集・集計処理を一箇所に集約する。

役割分離の方針:
  本ファイル      … 集計ロジックのみ（表示・HTML生成は行わない）
  x_results_public.py    … ユーザー向け表示（信頼性・透明性重視）
  x_results_developer.py … AI改善向け表示（特徴量・BuyScore分析等）
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

import x_release_storage

from x_verification import (
    load_today_records, load_records_range,
    aggregate, aggregate_by_rank,
    calc_brand_trust, analyze_misses,
    trend_vs_previous, _yesterday_jst, _safe_float,
)

JST = timezone(timedelta(hours=9))

HIT_RECORD_CSV   = "hit_record.csv"
DAILY_STATS_JSON = "daily_stats.json"
BUYSCORE_LOG     = "buyscore_log.jsonl"
BUYSCORE_CONFIG  = "buyscore_config.json"
ASAHI_CONFIG     = "asahi_config.json"


# ════════════════════════════════════════════════════════════
# 期間別データ集計（公開・開発共通）
# ════════════════════════════════════════════════════════════

def collect_all_periods(end_date: str) -> dict:
    """今日/7日/30日/累計の4期間すべてを集計してまとめて返す。"""
    records_1d  = load_today_records(end_date)
    records_7d  = load_records_range(days=7,  end_date=end_date)
    records_30d = load_records_range(days=30, end_date=end_date)
    records_all = load_records_range(days=None, end_date=end_date)

    periods = {}
    for label, records in [
        ("today", records_1d), ("d7", records_7d),
        ("d30", records_30d), ("all", records_all),
    ]:
        agg = aggregate(records)
        rank_data = aggregate_by_rank(records)
        periods[label] = {
            "records": records,
            "agg": agg,
            "rank_data": rank_data,
        }

    return periods


# ════════════════════════════════════════════════════════════
# daily_stats.json（新聞掲載件数）読み込み
# ════════════════════════════════════════════════════════════

def load_daily_stats(date_str: str) -> dict:
    """daily_stats.json から指定日の掲載件数・レース一覧を返す。"""
    x_release_storage.download_file(DAILY_STATS_JSON, DAILY_STATS_JSON)
    if not os.path.exists(DAILY_STATS_JSON):
        return {}
    try:
        with open(DAILY_STATS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(date_str, {})
    except Exception:
        return {}


def load_daily_stats_range(end_date: str, days: int) -> dict:
    """daily_stats.json から end_date を含む過去 days 日間の全日付分を返す。"""
    x_release_storage.download_file(DAILY_STATS_JSON, DAILY_STATS_JSON)
    if not os.path.exists(DAILY_STATS_JSON):
        return {}
    try:
        with open(DAILY_STATS_JSON, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except Exception:
        return {}

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=days - 1)
    start_str = start_dt.strftime("%Y%m%d")

    return {d: v for d, v in all_data.items() if start_str <= d <= end_date}


# ════════════════════════════════════════════════════════════
# ブランド別「掲載件数中Y件達成」集計
# ════════════════════════════════════════════════════════════

def calc_brand_results(records: list[dict], daily_stats: dict, brand: str) -> dict:
    """
    「掲載X件中、Y件が条件達成」を計算して返す。
    brand: "danger" / "manshuu" / "hot_high" / "korogashi"
    戻り値: {"listed":, "checked":, "hit":, "rate":, "has_data":}
    """
    if brand == "manshuu":
        listed_count = daily_stats.get("manshuu", {}).get("count", 0)
        check_races  = {
            (str(r["venue_num"]), str(r["race"]))
            for r in daily_stats.get("manshuu", {}).get("top10", [])
        }
    elif brand == "danger":
        listed_count = daily_stats.get("danger", {}).get("count", 0)
        check_races  = {
            (str(r["venue_num"]), str(r["race"]))
            for r in daily_stats.get("danger", {}).get("races", [])
        }
    else:
        listed_count = daily_stats.get("manshuu", {}).get("count", 0)
        check_races  = {
            (str(r["venue_num"]), str(r["race"]))
            for r in daily_stats.get("manshuu", {}).get("races", [])
        }

    if not check_races or not records:
        return {"listed": listed_count, "checked": 0, "hit": 0, "rate": 0.0, "has_data": False}

    checked = 0
    hit     = 0
    already_seen: set[tuple] = set()   # 同一レースの重複カウントを防ぐ
    for r in records:
        key = (str(r.get("venue_num", "")), str(r.get("race", "")))
        if key not in check_races:
            continue
        if key in already_seen:
            # 同じレースが hit_record.csv に複数行存在する場合、
            # 掲載件数を超えてカウントされてしまうため2件目以降はスキップする
            continue
        result_combo = r.get("result_combo", "")
        if not (result_combo and "-" in result_combo):
            continue
        payout = float(r.get("payout") or 0)
        already_seen.add(key)
        checked += 1
        if brand == "danger":
            if result_combo.split("-")[0].strip() != "1":
                hit += 1
        elif brand == "manshuu":
            if payout > 10000:
                hit += 1
        elif brand == "hot_high":
            if payout > 5000:
                hit += 1
        elif brand == "korogashi":
            if payout > 2700:
                hit += 1

    rate = round(hit / checked * 100, 1) if checked > 0 else 0.0
    return {
        "listed":   listed_count,
        "checked":  checked,
        "hit":      hit,
        "rate":     rate,
        "has_data": checked > 0,
    }


def calc_brand_results_range(records: list[dict], daily_stats_range: dict, brand: str) -> dict:
    """複数日分の daily_stats を使って期間集計する（30日実績用）。"""
    if not daily_stats_range:
        return {"listed": 0, "checked": 0, "hit": 0, "rate": 0.0, "has_data": False}

    records_by_date: dict[str, list[dict]] = {}
    for r in records:
        d = str(r.get("date", "")).replace("-", "")
        records_by_date.setdefault(d, []).append(r)

    total_listed = total_checked = total_hit = 0
    for date_str, daily_stats in daily_stats_range.items():
        day_records = records_by_date.get(date_str, [])
        result = calc_brand_results(day_records, daily_stats, brand)
        total_listed  += result["listed"]
        total_checked += result["checked"]
        total_hit     += result["hit"]

    rate = round(total_hit / total_checked * 100, 1) if total_checked > 0 else 0.0
    return {
        "listed": total_listed, "checked": total_checked,
        "hit": total_hit, "rate": rate, "has_data": total_checked > 0,
    }


# ════════════════════════════════════════════════════════════
# korogashi / hot_motor / awakening_motor の的中率
# ════════════════════════════════════════════════════════════
# 【既知の制約】激走・覚醒モーターは (venue_num, motor_no) 単位のデータで
# レース番号を持たないため、hit_record.csv との的中判定ができない。
# 転がし候補は生成元スクリプト未共有のため判定ロジック不明。
# いずれも "data_available": False を返し、呼び出し側は「データ整備中」と表示すること。

def calc_korogashi_results() -> dict:
    """転がし候補の的中率（未実装・データ整備中）"""
    return {"listed": 0, "hit": 0, "rate": 0.0, "data_available": False,
            "reason": "korogashi_cache.json 生成元スクリプト未確認のため未実装"}


def calc_hot_motor_results() -> dict:
    """激走モーターの的中率（未実装・データ整備中）"""
    return {"listed": 0, "hit": 0, "rate": 0.0, "data_available": False,
            "reason": "モーター単位データのためレース番号と紐付け不可"}


def calc_awakening_results() -> dict:
    """覚醒モーターの的中率（未実装・データ整備中）"""
    return {"listed": 0, "hit": 0, "rate": 0.0, "data_available": False,
            "reason": "モーター単位データのためレース番号と紐付け不可"}


# ════════════════════════════════════════════════════════════
# AIランキング払戻一覧
# ════════════════════════════════════════════════════════════

def calc_ranking_payouts(records: list[dict], daily_stats: dict, date_str: Optional[str] = None) -> list[dict]:
    """
    「今日のAIランキング」トップ10の各レースの実際の払戻・結果を返す。

    hit_record.csv にデータがあればそれを優先して使う。
    ない場合（AIランキング上位でも notify_arashi.py の通知基準に届かず
    未記録のレース）は BoatraceOpenAPI の results API から直接結果を
    取得して補完する（fetch_results_for_date を1日1回だけ呼び、
    複数レースの検索に使い回すことでAPI呼び出し回数を抑える）。

    date_str: 対象日 YYYYMMDD。省略時は records[0]["date"] から推測する
              （records が空の場合は API 補完ができない）。
    """
    ranking = daily_stats.get("ranking", [])
    if not ranking:
        return []

    record_map: dict[tuple, dict] = {}
    for r in records:
        key = (str(r.get("venue_num", "")), str(r.get("race", "")))
        record_map[key] = r

    # 対象日を特定（明示指定を優先、なければ records から推測）
    if not date_str and records:
        date_str = str(records[0].get("date", "")).replace("-", "")

    api_results_cache: Optional[dict] = None  # 遅延取得（1日分を1回だけ）

    result = []
    for item in ranking:
        key = (str(item.get("venue_num", "")), str(item.get("race", "")))
        rec = record_map.get(key)
        payout, result_combo, has_data = None, "", False

        if rec:
            result_combo = rec.get("result_combo", "")
            if result_combo and "-" in result_combo:
                payout = int(_safe_float(rec.get("payout")))
                has_data = True

        if not has_data and date_str:
            # hit_record.csv に記録がない → results API から直接補完する
            if api_results_cache is None:
                api_results_cache = fetch_results_for_date(date_str) or {}
            api_result = find_race_result(
                api_results_cache,
                int(item.get("venue_num", 0)),
                int(item.get("race", 0)) if str(item.get("race", "")).isdigit() else 0,
            )
            if api_result:
                result_combo = api_result["combo"]
                payout = api_result["payout"]
                has_data = True

        result.append({
            "venue": item.get("venue", ""), "race": item.get("race", ""),
            "match_index": item.get("match_index", 0),
            "payout": payout, "result_combo": result_combo, "has_data": has_data,
        })
    return result


# ════════════════════════════════════════════════════════════
# BoatraceOpenAPI results 直接取得（hit_record.csv未記録レースの補完用）
# ════════════════════════════════════════════════════════════

RESULTS_URL = "https://boatraceopenapi.github.io/results/v2"
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}
_HTTP_TIMEOUT = 15


def fetch_results_for_date(date_str: str) -> Optional[dict]:
    """
    指定日1日分のレース結果をBoatraceOpenAPIから1回だけ取得する。
    複数レースの結果検索に使い回すことで、レースごとにAPIを叩く
    非効率を避ける。失敗時は None を返す。
    """
    url = f"{RESULTS_URL}/{date_str[:4]}/{date_str}.json"
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        # 前日分が today.json 側にしかない場合のフォールバック
        try:
            today_str = datetime.now(JST).strftime("%Y%m%d")
            if date_str == today_str:
                r = requests.get(f"{RESULTS_URL}/today.json", headers=_HTTP_HEADERS, timeout=_HTTP_TIMEOUT)
                r.raise_for_status()
                return r.json()
        except requests.RequestException:
            pass
        return None


def find_race_result(results_data: dict, venue_num: int, race_number: int) -> Optional[dict]:
    """
    fetch_results_for_date() が返した1日分の結果データから、
    指定レースの3連単結果・払戻を検索する。
    戻り値: {"combo": "2-3-1", "payout": 4520} or None
    """
    if not results_data:
        return None

    for r in results_data.get("results", []):
        if (r.get("race_stadium_number") == venue_num
                and r.get("race_number") == race_number):
            combo, payout = "不明", 0
            payouts = r.get("payouts", {})
            if isinstance(payouts, dict):
                trifecta = payouts.get("trifecta", [])
                if isinstance(trifecta, list) and trifecta:
                    try:
                        combo  = trifecta[0].get("combination", "不明")
                        payout = int(trifecta[0].get("payout", 0))
                    except (ValueError, TypeError):
                        pass

            if combo == "不明":
                boats = r.get("boats", [])
                if isinstance(boats, dict):
                    boats = list(boats.values())
                order = sorted(
                    [b for b in boats if isinstance(b, dict) and b.get("racer_place_number")],
                    key=lambda b: b.get("racer_place_number", 99),
                )
                if len(order) >= 3:
                    combo = "-".join(str(b.get("racer_boat_number", "?")) for b in order[:3])

            if combo == "不明":
                return None
            return {"combo": combo, "payout": payout}

    return None


# ════════════════════════════════════════════════════════════
# 全体回収率・ROI・期間平均
# ════════════════════════════════════════════════════════════

def calc_overall_roi(records: list[dict]) -> dict:
    """
    【購入判定分離】実際に購入対象(purchased=1)となった買い目のみで
    回収率・ROIを算出する。「予想はしたが見送った」レコード(purchased=0)は
    投資額0円のため、含めても含めなくても計算結果は変わらないが、
    件数集計を正確にするため明示的に除外する。

    回収率 = 払戻金合計 ÷ 投資金額合計 × 100
    ROI    = (払戻金合計 − 投資金額合計) ÷ 投資金額合計 × 100
    """
    purchased = [
        r for r in records
        if r.get("pred_combo")
        and r.get("result_combo") and "-" in r.get("result_combo", "")
        and int(_safe_float(r.get("purchased"), 1)) == 1   # 旧データ(purchased列なし)は1扱い
    ]
    if not purchased:
        return {
            "recovery_rate": 0.0, "roi": 0.0, "total_cost": 0, "total_return": 0,
            "n": 0, "n_races": 0, "n_bets": 0,
        }

    # 【バグ修正】旧実装は "int(_safe_float(cost)) or 100" となっており、
    # cost=0（投資なし）が Python の or 演算子で 100 に化けてしまっていた。
    # cost は明示的にそのまま使う（0は0のまま扱う）。
    total_cost = sum(int(_safe_float(r.get("cost"), 0)) for r in purchased)
    total_return = 0
    for r in purchased:
        if int(_safe_float(r.get("hit"))) == 1:
            total_return += int(_safe_float(r.get("payout")))

    recovery_rate = round(total_return / total_cost * 100, 1) if total_cost > 0 else 0.0
    roi = round((total_return - total_cost) / total_cost * 100, 1) if total_cost > 0 else 0.0
    total_bets = sum(int(_safe_float(r.get("n_bets"), 1)) for r in purchased)
    return {
        "recovery_rate": recovery_rate, "roi": roi,
        "total_cost": total_cost, "total_return": total_return,
        "n": len(purchased),        # 購入対象レース数
        "n_races": len(purchased),  # 明示的なエイリアス（表示用）
        "n_bets": total_bets,       # 購入点数（1レースで複数点購入した場合を含む）
    }


# ════════════════════════════════════════════════════════════
# MVP予想・惜しかった予想（公開用ページ向け）
# ════════════════════════════════════════════════════════════

def find_mvp_prediction(records: list[dict]) -> Optional[dict]:
    """
    「昨日のMVP予想」= 実際に購入(purchased=1)して的中した予想の中で
    最も払戻が高かったもの。的中がなければ None を返す。
    """
    hits = [
        r for r in records
        if int(_safe_float(r.get("hit"))) == 1
        and int(_safe_float(r.get("purchased"), 1)) == 1
    ]
    if not hits:
        return None
    best = max(hits, key=lambda r: _safe_float(r.get("payout")))
    return {
        "venue": best.get("venue", ""), "race": best.get("race", ""),
        "pred_combo": best.get("pred_combo", ""), "result_combo": best.get("result_combo", ""),
        "payout": int(_safe_float(best.get("payout"))),
        "why_bet": best.get("why_bet", ""),
    }


def find_close_misses(records: list[dict], limit: int = 3) -> list[dict]:
    """
    「昨日の惜しかった予想」= 外れたレースのうち、荒れスコアが高かった
    （＝AIが自信を持って外した）順に最大 limit 件を返す。
    x_verification.analyze_misses() を再利用する。
    """
    miss_analysis = analyze_misses(records, top_n=limit)
    return miss_analysis.get("top_misses", [])


def generate_ai_comment_for_miss(miss: dict) -> str:
    """外れレースに対する簡潔なAIコメントを生成する（「○○が想定外でした」形式）。"""
    reason = miss.get("reason", "")
    if "1残り" in reason:
        return "1号艇残りが想定外でした"
    if "万舟" in reason:
        return "荒れの度合いが想定を上回りました"
    if "展開違い" in reason:
        return "レース展開が想定と異なりました"
    if "着順違い" in reason:
        return "上位艇の入れ替わりが想定外でした"
    return "結果が想定と異なりました"


# ════════════════════════════════════════════════════════════
# 前日比較
# ════════════════════════════════════════════════════════════

def calc_day_over_day_comparison(date_str: str) -> dict:
    """
    危険艇・万舟の的中率について、前日比較を返す。
    戻り値: {"danger": {"yesterday": 68.0, "today": 72.0, "diff": 4.0}, "manshuu": {...}}
    """
    today_dt = datetime.strptime(date_str, "%Y%m%d")
    prev_date = (today_dt - timedelta(days=1)).strftime("%Y%m%d")

    today_stats = load_daily_stats(date_str)
    prev_stats  = load_daily_stats(prev_date)

    today_records = load_today_records(date_str)
    prev_records  = load_today_records(prev_date)

    result = {}
    for brand in ["danger", "manshuu"]:
        today_r = calc_brand_results(today_records, today_stats, brand)
        prev_r  = calc_brand_results(prev_records, prev_stats, brand)
        result[brand] = {
            "yesterday": prev_r["rate"] if prev_r["has_data"] else None,
            "today":     today_r["rate"] if today_r["has_data"] else None,
            "diff": (
                round(today_r["rate"] - prev_r["rate"], 1)
                if today_r["has_data"] and prev_r["has_data"] else None
            ),
        }
    return result


# ════════════════════════════════════════════════════════════
# 【開発用】特徴量分析
# ════════════════════════════════════════════════════════════

FEATURE_COLUMNS = [
    "feat_win_rate", "feat_motor", "feat_avg_st",
    "feat_course_st_1c", "feat_course_rank_1c",
]


def analyze_features(records: list[dict], min_samples: int = 30) -> dict:
    """
    特徴量ごとの平均値・分散・的中との相関を集計する（開発用ページ向け）。
    サンプル不足の場合は "data_available": False を返す。
    """
    valid_records = [
        r for r in records
        if r.get("feat_win_rate") not in (None, "", "None")
    ]
    if len(valid_records) < min_samples:
        return {
            "data_available": False,
            "n_samples": len(valid_records),
            "min_required": min_samples,
            "reason": f"サンプル不足（{len(valid_records)}件 < {min_samples}件）。特徴量記録開始直後のため蓄積中。",
        }

    result = {"data_available": True, "n_samples": len(valid_records), "features": {}}
    hits = [int(_safe_float(r.get("hit"))) for r in valid_records]

    for col in FEATURE_COLUMNS:
        vals = []
        for r in valid_records:
            v = r.get(col)
            if v not in (None, "", "None"):
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if len(vals) < 2:
            continue
        mean = statistics.mean(vals)
        variance = statistics.variance(vals)

        # 的中との相関（サンプル同数の場合のみ）
        paired = [
            (float(r[col]), int(_safe_float(r.get("hit"))))
            for r in valid_records
            if r.get(col) not in (None, "", "None")
        ]
        corr = None
        if len(paired) >= 10:
            xs = [p[0] for p in paired]
            ys = [p[1] for p in paired]
            try:
                corr = round(statistics.correlation(xs, ys), 3)
            except (statistics.StatisticsError, ZeroDivisionError):
                corr = None

        result["features"][col] = {
            "mean": round(mean, 4), "variance": round(variance, 4),
            "n": len(vals), "correlation_with_hit": corr,
        }

    return result


def rank_improvement_candidates(feature_analysis: dict, top_n: int = 5) -> list[dict]:
    """
    「現在最も改善効果が高い特徴量」TOP Nを、的中との相関の絶対値順に返す。
    特徴量分析が未実施の場合は空リストを返す。
    """
    if not feature_analysis.get("data_available"):
        return []
    candidates = []
    for name, stats in feature_analysis.get("features", {}).items():
        corr = stats.get("correlation_with_hit")
        if corr is not None:
            candidates.append({"feature": name, "correlation": corr, "abs_correlation": abs(corr)})
    candidates.sort(key=lambda x: -x["abs_correlation"])
    return candidates[:top_n]


# ════════════════════════════════════════════════════════════
# 【開発用】外れ理由分類
# ════════════════════════════════════════════════════════════

def classify_miss_reasons(records: list[dict]) -> dict:
    """
    外れたレースを9カテゴリに分類し、件数ランキングを返す。
    カテゴリ: 本命決着・穴決着・モーター評価ミス・選手評価ミス・
              コース評価ミス・買い目構成ミス・高期待値だが低確率・
              データ不足・その他
    """
    misses = [
        r for r in records
        if r.get("pred_combo") and int(_safe_float(r.get("hit"), 1)) == 0
        and r.get("result_combo") and "-" in r.get("result_combo", "")
    ]
    if not misses:
        return {"total": 0, "categories": {}}

    categories: dict[str, int] = {}
    for r in misses:
        cat = _classify_single_miss(r)
        categories[cat] = categories.get(cat, 0) + 1

    ranking = sorted(categories.items(), key=lambda x: -x[1])
    return {
        "total": len(misses),
        "categories": dict(ranking),
        "top_reason": ranking[0][0] if ranking else None,
    }


MISS_CATEGORIES = [
    "本命決着（荒れを読み過ぎた）",
    "穴決着（本命を評価し過ぎた）",
    "モーター評価ミス",
    "選手評価ミス",
    "コース評価ミス",
    "買い目構成ミス",
    "高期待値だが低確率",
    "データ不足",
    "その他",
]


def _feat_float(r: dict, key: str) -> Optional[float]:
    """feat_* 列を安全にfloat変換する。空・None・"None"はNoneを返す。"""
    v = r.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _classify_single_miss(r: dict) -> str:
    """
    1件の外れレコードを9カテゴリのいずれかに分類する（優先順位付きルールベース）。

    判定順序:
      1. データ不足        … 分類に必要な特徴量が揃っていない
      2. 本命決着          … 荒れ予想(upset_score高)だったのに1号艇が普通に1着
      3. 穴決着            … 本命予想(upset_score低 or 1号艇軸)だったのに荒れた
      4. 買い目構成ミス    … 1着は的中、2-3着の組み合わせのみ外れ
      5. モーター評価ミス  … モーター高評価の軸艇が来なかった
      6. 選手評価ミス      … 勝率高評価の軸艇が来なかった
      7. コース評価ミス    … コース別ST実績が良い評価の軸艇が来なかった
      8. 高期待値だが低確率 … EVは高いが的中確率自体が低い買い目だった
      9. その他            … 上記いずれにも該当しない
    """
    result_combo = r.get("result_combo", "")
    pred_combo   = r.get("pred_combo", "")
    first_actual = result_combo.split("-")[0].strip() if "-" in result_combo else ""
    first_pred   = pred_combo.split("-")[0].strip()   if "-" in pred_combo   else ""

    upset_score = _feat_float(r, "upset_score")
    pred_ev     = _feat_float(r, "pred_ev")
    pred_prob   = _feat_float(r, "pred_prob")
    feat_motor        = _feat_float(r, "feat_motor")
    feat_win_rate     = _feat_float(r, "feat_win_rate")
    feat_course_st    = _feat_float(r, "feat_course_st_1c")
    feat_course_rank  = _feat_float(r, "feat_course_rank_1c")

    # ── 1. データ不足 ────────────────────────────────────────
    # 荒れ判定・特徴量のいずれも記録されていなければ分類材料がない
    if upset_score is None and feat_motor is None and feat_win_rate is None:
        return "データ不足"

    # ── 2. 本命決着（荒れを読み過ぎた） ──────────────────────
    if upset_score is not None and upset_score >= 6.0 and first_actual == "1":
        return "本命決着（荒れを読み過ぎた）"

    # ── 3. 穴決着（本命を評価し過ぎた） ──────────────────────
    if first_actual != "1" and (
        (upset_score is not None and upset_score < 4.0) or first_pred == "1"
    ):
        return "穴決着（本命を評価し過ぎた）"

    # ── 4. 買い目構成ミス（1着は的中、2-3着のみ外れ） ────────
    if first_pred and first_pred == first_actual:
        return "買い目構成ミス"

    # ── 5. モーター評価ミス ──────────────────────────────────
    if feat_motor is not None and feat_motor >= 38 and first_pred != first_actual:
        return "モーター評価ミス"

    # ── 6. 選手評価ミス ──────────────────────────────────────
    if feat_win_rate is not None and feat_win_rate >= 6.0 and first_pred != first_actual:
        return "選手評価ミス"

    # ── 7. コース評価ミス ────────────────────────────────────
    if (
        (feat_course_st is not None and feat_course_st <= 0.15)
        or (feat_course_rank is not None and feat_course_rank <= 2.0)
    ) and first_pred != first_actual:
        return "コース評価ミス"

    # ── 8. 高期待値だが低確率 ────────────────────────────────
    if pred_ev is not None and pred_prob is not None and pred_ev >= 2.0 and pred_prob < 0.03:
        return "高期待値だが低確率"

    return "その他"


# ════════════════════════════════════════════════════════════
# 【開発用】艇番評価バイアス分析
# ════════════════════════════════════════════════════════════

def analyze_boat_number_bias(records: list[dict]) -> dict:
    """
    予想1着艇番と実際の1着艇番の分布を比較し、AIが特定の艇番を
    過大評価・過小評価していないかを分析する。

    戻り値:
      {
        "data_available": bool,
        "n": int,
        "pred_distribution":   {"1": 45.2, "2": 15.0, ...},  # 予想1着に選んだ割合(%)
        "actual_distribution": {"1": 40.1, "2": 18.3, ...},  # 実際1着だった割合(%)
        "bias": {"1": +5.1, "2": -3.3, ...},  # pred - actual（正=過大評価、負=過小評価）
        "most_overestimated":  {"lane": "1", "diff": 5.1},
        "most_underestimated": {"lane": "6", "diff": -4.2},
      }
    """
    valid = [
        r for r in records
        if r.get("pred_combo") and "-" in r.get("pred_combo", "")
        and r.get("result_combo") and "-" in r.get("result_combo", "")
    ]
    if len(valid) < 10:
        return {"data_available": False, "n": len(valid), "reason": "サンプル不足（10件未満）"}

    pred_count   = {str(i): 0 for i in range(1, 7)}
    actual_count = {str(i): 0 for i in range(1, 7)}

    for r in valid:
        p_first = r["pred_combo"].split("-")[0].strip()
        a_first = r["result_combo"].split("-")[0].strip()
        if p_first in pred_count:
            pred_count[p_first] += 1
        if a_first in actual_count:
            actual_count[a_first] += 1

    n = len(valid)
    pred_dist   = {k: round(v / n * 100, 1) for k, v in pred_count.items()}
    actual_dist = {k: round(v / n * 100, 1) for k, v in actual_count.items()}
    bias        = {k: round(pred_dist[k] - actual_dist[k], 1) for k in pred_dist}

    most_over  = max(bias.items(), key=lambda x: x[1])
    most_under = min(bias.items(), key=lambda x: x[1])

    return {
        "data_available": True,
        "n": n,
        "pred_distribution": pred_dist,
        "actual_distribution": actual_dist,
        "bias": bias,
        "most_overestimated":  {"lane": most_over[0],  "diff": most_over[1]},
        "most_underestimated": {"lane": most_under[0], "diff": most_under[1]},
    }


# ════════════════════════════════════════════════════════════
# 【開発用】本日の改善候補（統計ベース自動生成）
# ════════════════════════════════════════════════════════════

def generate_improvement_suggestions(
    miss_classification: dict,
    boat_bias: dict,
    top_n: int = 3,
) -> list[str]:
    """
    外れ理由ランキング・艇番評価バイアスの統計結果から、
    「本日の改善候補」を動的に生成する（固定文は使用しない）。
    """
    candidates: list[tuple[float, str]] = []  # (優先度スコア, 文言)

    categories = miss_classification.get("categories", {})
    total = miss_classification.get("total", 0)

    # ── カテゴリ別外れ件数に基づく提案 ──────────────────────
    category_suggestions = {
        "本命決着（荒れを読み過ぎた）":
            lambda n, pct: f"荒れ判定の閾値をやや引き上げる（本命決着が{n}件・外れの{pct:.0f}%で最多、荒れを読み過ぎる傾向）",
        "穴決着（本命を評価し過ぎた）":
            lambda n, pct: f"1号艇の信頼度評価をやや下げる（穴決着が{n}件・外れの{pct:.0f}%、本命評価が過大）",
        "モーター評価ミス":
            lambda n, pct: f"モーター評価の重みを見直す（モーター評価ミスが{n}件・外れの{pct:.0f}%）",
        "選手評価ミス":
            lambda n, pct: f"全国勝率評価の重みを見直す（選手評価ミスが{n}件・外れの{pct:.0f}%）",
        "コース評価ミス":
            lambda n, pct: f"コース別ST実績の重みを見直す（コース評価ミスが{n}件・外れの{pct:.0f}%）",
        "買い目構成ミス":
            lambda n, pct: f"フォーメーション（2〜3着の相手選定）ロジックを見直す（買い目構成ミスが{n}件・外れの{pct:.0f}%、軸自体は的中）",
        "高期待値だが低確率":
            lambda n, pct: f"高配当補正（EV偏重）をやや弱める（高EV低確率の外れが{n}件・外れの{pct:.0f}%）",
    }

    for cat, count in categories.items():
        if cat in ("データ不足", "その他"):
            continue
        pct = count / total * 100 if total else 0
        if cat in category_suggestions and count >= 2:
            # 件数と割合を優先度スコアとする（多いほど優先）
            priority = count + pct / 10
            candidates.append((priority, category_suggestions[cat](count, pct)))

    # ── 艇番評価バイアスに基づく提案 ──────────────────────────
    if boat_bias.get("data_available"):
        over  = boat_bias["most_overestimated"]
        under = boat_bias["most_underestimated"]
        if over["diff"] >= 3.0:
            pred_pct = boat_bias["pred_distribution"][over["lane"]]
            actual_pct = boat_bias["actual_distribution"][over["lane"]]
            candidates.append((
                over["diff"] * 2,
                f"{over['lane']}号艇評価をやや下げる"
                f"（予想1着率{pred_pct}% vs 実際{actual_pct}%、+{over['diff']}ptの過大評価）",
            ))
        if under["diff"] <= -3.0:
            pred_pct = boat_bias["pred_distribution"][under["lane"]]
            actual_pct = boat_bias["actual_distribution"][under["lane"]]
            candidates.append((
                abs(under["diff"]) * 2,
                f"{under['lane']}号艇評価をやや上げる"
                f"（予想1着率{pred_pct}% vs 実際{actual_pct}%、{under['diff']}ptの過小評価）",
            ))

    # 優先度スコア降順でソートし、上位N件のテキストのみ返す
    candidates.sort(key=lambda x: -x[0])
    return [text for _, text in candidates[:top_n]]


# ════════════════════════════════════════════════════════════
# 【開発用】BuyScore分析
# ════════════════════════════════════════════════════════════

def analyze_buyscore_log(days: int = 7) -> dict:
    """
    buyscore_log.jsonl からスコア分布・購入/見送り件数を集計する。
    ファイルがない場合は data_available=False を返す。
    """
    x_release_storage.download_file(BUYSCORE_LOG, BUYSCORE_LOG)
    if not os.path.exists(BUYSCORE_LOG):
        return {"data_available": False, "reason": "buyscore_log.jsonl が見つかりません"}

    cutoff = (datetime.now(JST) - timedelta(days=days)).strftime("%Y%m%d")
    entries = []
    try:
        with open(BUYSCORE_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("date", "") >= cutoff:
                        entries.append(e)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return {"data_available": False, "reason": "読み込みエラー"}

    if not entries:
        return {"data_available": False, "reason": "対象期間のログなし", "n_races": 0}

    total_races = len(entries)
    passthrough_count = sum(1 for e in entries if e.get("passthrough_reason"))
    buy_count = total_races - passthrough_count

    score_buckets: dict[str, int] = {}
    for e in entries:
        for c in e.get("candidates", []):
            bs = c.get("buyscore")
            if bs is None:
                continue
            bucket = f"{int(bs // 10) * 10}-{int(bs // 10) * 10 + 9}"
            score_buckets[bucket] = score_buckets.get(bucket, 0) + 1

    return {
        "data_available": True,
        "n_races": total_races,
        "buy_count": buy_count,
        "passthrough_count": passthrough_count,
        "passthrough_rate": round(passthrough_count / total_races * 100, 1) if total_races else 0,
        "score_distribution": dict(sorted(score_buckets.items())),
    }


# ════════════════════════════════════════════════════════════
# 【開発用】学習履歴
# ════════════════════════════════════════════════════════════

def get_learning_history() -> dict:
    """
    buyscore_config.json / asahi_config.json から学習実行履歴を取得する。
    Phase1時点では asahi_config は初期重みのまま（学習未実行）。
    """
    history = {"buyscore_tuning": None, "asahi_model": None}

    if os.path.exists(BUYSCORE_CONFIG):
        try:
            with open(BUYSCORE_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            tuning = cfg.get("tuning", {})
            if tuning.get("last_tuned"):
                history["buyscore_tuning"] = {
                    "last_tuned":   tuning.get("last_tuned"),
                    "last_samples": tuning.get("last_samples"),
                }
        except Exception:
            pass

    if os.path.exists(ASAHI_CONFIG):
        try:
            with open(ASAHI_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            history["asahi_model"] = {
                "model_version": cfg.get("model_version"),
                "phase":         cfg.get("phase"),
                "generated_at":  cfg.get("generated_at"),
                "note":          cfg.get("note"),
            }
        except Exception:
            pass

    return history


# ════════════════════════════════════════════════════════════
# 【開発用】ブランド別 件数・的中率・回収率・ROI
# ════════════════════════════════════════════════════════════

def calc_brand_dev_stats(records: list[dict], daily_stats: dict) -> dict:
    """
    危険艇・万舟・転がし・激走・覚醒それぞれについて、
    件数・的中率・回収率・ROIをまとめて返す（開発用ページ向け）。
    """
    result = {}

    for brand in ["danger", "manshuu"]:
        br = calc_brand_results(records, daily_stats, brand)
        brand_records = [
            r for r in records
            if (str(r.get("venue_num","")), str(r.get("race",""))) in {
                (str(x["venue_num"]), str(x["race"]))
                for x in daily_stats.get(brand if brand == "danger" else "manshuu", {}).get(
                    "races" if brand == "danger" else "top10", []
                )
            }
        ]
        roi = calc_overall_roi(brand_records)
        result[brand] = {**br, "recovery_rate": roi["recovery_rate"], "roi": roi["roi"]}

    for brand, fn in [
        ("korogashi", calc_korogashi_results),
        ("hot_motor", calc_hot_motor_results),
        ("awakening_motor", calc_awakening_results),
    ]:
        result[brand] = fn()

    return result
