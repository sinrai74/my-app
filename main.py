"""
main.py  ── CLI エントリーポイント

使い方:
  # モデル学習
  python main.py train --data ./data

  # 今日の予想
  python main.py predict --data ./data --date 2026-04-14

  # 結果照合
  python main.py result --data ./data --date 2026-04-14

  # 気象オプション付き予想
  python main.py predict --data ./data --date 2026-04-14 \\
      --wind-speed 3.0 --wind-dir 追 --tide 満

  # 全場設定の比較表表示
  python main.py config
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── 自モジュール ──────────────────────────────────────────────
from config import BASE_CONFIG, get_venue_config, print_config_comparison
from parsers import (
    parse_k_file, parse_fan_files, parse_bangumi,
    merge_bangumi_fan, attach_k_results, discover_files,
)
from features import (
    engineer_features, build_targets, recalc_flags,
    compute_all_dynamic_ev,
)
from models import (
    fit_models, predict_all, timeseries_cv,
    save_model, load_model,
)
from betting import (
    generate_bets, simulate_returns,
    build_race_picks, build_detail_df,
)
from output import save_picks_excel, save_result_excel, print_summary


# ════════════════════════════════════════════════════════════
# サブコマンド: config
# ════════════════════════════════════════════════════════════

def cmd_config(args) -> None:
    """全場設定の比較表を表示する"""
    print_config_comparison()


# ════════════════════════════════════════════════════════════
# サブコマンド: train
# ════════════════════════════════════════════════════════════

def cmd_train(args) -> None:
    """
    過去の番組表(B) + K結果 + fan で学習し model.pkl を保存する。
    """
    data_dir  = Path(args.data)
    model_out = Path(args.model)

    files = discover_files(data_dir)
    print(f"[train] データディレクトリ: {data_dir}")
    print(f"  fan       : {[f.name for f in files['fan']]}")
    print(f"  番組表(過去): {[f.name for f in files['bangumi_past']]}")
    print(f"  K結果     : {[f.name for f in files['k_files']]}")

    if not files["k_files"]:
        print("❌ Kファイルが見つかりません")
        sys.exit(1)

    learn_bangumi = files["bangumi_past"] or (
        [files["bangumi_today"]] if files["bangumi_today"] else [])
    if not learn_bangumi:
        print("❌ 番組表ファイルが見つかりません")
        sys.exit(1)

    # データ読み込み
    fan_df  = parse_fan_files(files["fan"])
    past_bg = pd.concat([parse_bangumi(f) for f in learn_bangumi], ignore_index=True)
    k_all   = pd.concat([parse_k_file(f)  for f in files["k_files"]], ignore_index=True)

    print(f"\n学習データ: 番組表={len(past_bg)}行 / K結果={len(k_all)}R")

    # 動的 EV_MIN
    print("\n━━ 動的EV_MIN計算 ━━")
    dynamic_ev_map = compute_all_dynamic_ev(k_all)

    # 特徴量生成
    merged = merge_bangumi_fan(past_bg, fan_df)
    merged = attach_k_results(merged, k_all)
    print("特徴量生成中...")
    merged = engineer_features(
        merged,
        wind_speed=0.0, wind_direction="", tide_level="",
        dynamic_ev_map=dynamic_ev_map,
    )
    merged = build_targets(merged)

    # CV + 最終学習
    cv_res  = timeseries_cv(merged, fast_mode=not args.full_cv)
    models  = cv_res["final_models"]
    metrics = cv_res["metrics_df"]

    # CV 回収率確認
    if len(cv_res["pred_df"]) > 0:
        bets_cv = generate_bets(cv_res["pred_df"])
        sim_cv  = simulate_returns(cv_res["pred_df"], bets_cv)
        print(f"\n━━ CV回収率 ━━")
        print(f"  {sim_cv['レース数']}R / 回収率{sim_cv['回収率']}% / 的中率{sim_cv['的中率']}%")
        print(f"  ¥10,000 → ¥{sim_cv['最終残高']:,.0f}")

    # 保存
    save_model(
        models, model_out,
        extra={
            "dynamic_ev_map": dynamic_ev_map,
            "base_config":    BASE_CONFIG,
            "cv_metrics":     metrics,
        }
    )


# ════════════════════════════════════════════════════════════
# サブコマンド: predict
# ════════════════════════════════════════════════════════════

def cmd_predict(args) -> None:
    """
    今日の番組表(B) と学習済みモデルで予想を生成し Excel を出力する。
    """
    data_dir   = Path(args.data)
    model_path = Path(args.model)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    race_date  = datetime.strptime(args.date, "%Y-%m-%d").date()
    date_str   = race_date.strftime("%Y%m%d")
    output_file = out_dir / f"picks_{date_str}.xlsx"

    files = discover_files(data_dir)

    if not model_path.exists():
        print(f"❌ {model_path} が見つかりません → train を先に実行してください")
        sys.exit(1)

    model_data     = load_model(model_path)
    models         = model_data["models"]
    dynamic_ev_map = model_data.get("dynamic_ev_map", {})
    print(f"動的EV_MINマップ: {dynamic_ev_map}")

    fan_df = parse_fan_files(files["fan"])

    today_file = files["bangumi_today"]
    if today_file is None:
        print("❌ 今日の番組表が見つかりません")
        sys.exit(1)

    bg_today     = parse_bangumi(today_file)
    merged_today = merge_bangumi_fan(bg_today, fan_df)

    # 結果列を空で補完（予想時は不要）
    for col, dt in [("1着フラグ","Int64"),("3着内フラグ","Int64"),("着順","Int64")]:
        merged_today[col] = pd.array([None]*len(merged_today), dtype=dt)
    merged_today["返還フラグ"] = 0
    merged_today["人気"]  = np.nan
    merged_today["払戻"]  = np.nan

    print(f"特徴量生成中（風:{args.wind_dir or 'なし'}{args.wind_speed}m/s 潮:{args.tide or 'なし'}）...")
    merged_today = engineer_features(
        merged_today,
        wind_speed=args.wind_speed,
        wind_direction=args.wind_dir,
        tide_level=args.tide,
        dynamic_ev_map=dynamic_ev_map,
    )

    print("予測実行中...")
    pred_df = predict_all(models, merged_today)
    pred_df = recalc_flags(pred_df)

    picks_list = build_race_picks(pred_df)
    picks_df   = pd.DataFrame(picks_list)
    detail_df  = build_detail_df(pred_df)

    buy_n = picks_df["判定"].str.startswith("✅").sum()
    print(f"\n━━ 予想結果 ━━")
    print(f"  場数: {pred_df['場名'].nunique()} / "
          f"レース: {pred_df.groupby(['場コード','レースNo']).ngroups}")
    print(f"  ✅ 買い: {buy_n}R / "
          f"❌ 見送り: {picks_df['判定'].str.startswith('❌').sum()}R")
    print(f"  ⚠️ イン崩壊: {(picks_df.get('イン崩壊','') == '⚠️').sum()}R")
    print(f"  🌙 ナイター: {(picks_df.get('ナイター','') == '🌙').sum()}R")
    if "バリュースコア" in picks_df.columns:
        from config import VALUE_THRESHOLD
        print(f"  💰 バリュー(>{VALUE_THRESHOLD}): "
              f"{(picks_df['バリュースコア'] >= VALUE_THRESHOLD).sum()}R")

    show = ["場名","レースNo","軸艇番","軸選手","軸確率","軸真期待値",
            "バリュースコア","動的EV_MIN","ナイター","3連単①","3連単②","注意フラグ"]
    show = [c for c in show if c in picks_df.columns]
    print("\n━━ ✅ 買いレース（上位15） ━━")
    print(picks_df[picks_df["判定"].str.startswith("✅")][show].head(15).to_string(index=False))

    save_picks_excel(picks_df, detail_df, output_file)
    print(f"\n出力: {output_file}")


# ════════════════════════════════════════════════════════════
# サブコマンド: result
# ════════════════════════════════════════════════════════════

def cmd_result(args) -> None:
    """
    K結果ファイルと予想モデルを照合し結果Excelを出力する。
    """
    data_dir   = Path(args.data)
    model_path = Path(args.model)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    race_date   = datetime.strptime(args.date, "%Y-%m-%d").date()
    date_str    = race_date.strftime("%Y%m%d")
    output_file = out_dir / f"result_{date_str}.xlsx"

    files = discover_files(data_dir)

    if not model_path.exists():
        print(f"❌ {model_path} が見つかりません → train を先に実行してください")
        sys.exit(1)
    if not files["k_files"]:
        print("❌ Kファイルが見つかりません")
        sys.exit(1)

    model_data     = load_model(model_path)
    models         = model_data["models"]
    dynamic_ev_map = model_data.get("dynamic_ev_map", {})

    fan_df = parse_fan_files(files["fan"])
    today_file = files["bangumi_today"]
    if today_file is None:
        print("❌ 今日の番組表が見つかりません")
        sys.exit(1)

    bg_today     = parse_bangumi(today_file)
    merged_today = merge_bangumi_fan(bg_today, fan_df)
    for col, dt in [("1着フラグ","Int64"),("3着内フラグ","Int64"),("着順","Int64")]:
        merged_today[col] = pd.array([None]*len(merged_today), dtype=dt)
    merged_today["返還フラグ"] = 0
    merged_today["人気"]  = np.nan
    merged_today["払戻"]  = np.nan

    merged_today = engineer_features(
        merged_today,
        wind_speed=args.wind_speed,
        wind_direction=args.wind_dir,
        tide_level=args.tide,
        dynamic_ev_map=dynamic_ev_map,
    )
    pred_df = predict_all(models, merged_today)
    pred_df = recalc_flags(pred_df)

    # K結果を付与
    k_today     = parse_k_file(files["k_files"][-1])
    pred_result = attach_k_results(pred_df, k_today)
    pred_result["1着フラグ"]   = (pred_result["着順"] == 1).astype("Int64")
    pred_result["3着内フラグ"] = (pred_result["着順"] <= 3).astype("Int64")
    pred_result = recalc_flags(pred_result)

    # 的中集計
    total_bet = total_ret = 0.0
    results: list[dict] = []
    for (vc, rno), grp in pred_result.groupby(["場コード","レースNo"]):
        if grp["返還フラグ"].max() == 1:
            continue
        cfg    = get_venue_config(vc)
        ev_min = float(grp["動的EV_MIN"].iloc[0]) if "動的EV_MIN" in grp.columns \
                 else cfg.get("EV_MIN", BASE_CONFIG["EV_MIN"])
        p_min  = cfg.get("P_MIN", BASE_CONFIG["P_MIN"])
        grp    = grp.sort_values("アンサンブルスコア_IN順位")
        cands  = grp[(grp["予測_1着確率"] >= p_min) & (grp["真期待値"] >= ev_min)]
        if len(cands) == 0:
            continue
        axis    = cands.sort_values("真期待値", ascending=False).iloc[0]
        axis_no = int(axis["艇番"])
        bet     = 100.0
        total_bet += bet
        wr = grp[grp["着順"] == 1]
        if len(wr) == 0:
            continue
        winner   = int(wr["艇番"].values[0])
        haraisho = wr["払戻"].values[0]
        hit = (winner == axis_no) and pd.notna(haraisho)
        ret = float(haraisho) * (bet / 100) if hit else 0.0
        total_ret += ret
        results.append({
            "場名":        grp["場名"].iloc[0],
            "場コード":    vc,
            "レースNo":    rno,
            "軸艇番":      axis_no,
            "実際1着":     winner,
            "的中":        hit,
            "払戻":        haraisho,
            "回収":        ret,
            "真期待値":    float(axis["真期待値"]),
            "予測確率":    float(axis["予測_1着確率"]),
            "バリュースコア": float(axis.get("バリュースコア", 0)),
            "動的EV_MIN":  float(ev_min),
        })

    result_df = pd.DataFrame(results)
    print_summary(result_df, pred_result, total_bet, total_ret)
    save_result_excel(result_df, pred_result, output_file)
    print(f"\n出力: {output_file}")


# ════════════════════════════════════════════════════════════
# CLI パーサー
# ════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ボートレース予想エンジン（全24場統合版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 共通オプション
    parser.add_argument("--model", default="model.pkl",
                        help="モデルファイルパス (デフォルト: model.pkl)")
    parser.add_argument("--out",   default="./output",
                        help="Excel出力先ディレクトリ (デフォルト: ./output)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── config ──────────────────────────────────────────────
    sub.add_parser("config", help="全場設定の比較表を表示")

    # ── train ───────────────────────────────────────────────
    p_train = sub.add_parser("train", help="モデル学習")
    p_train.add_argument("--data", required=True,
                         help="fan/B/K ファイルが置かれたディレクトリ")
    p_train.add_argument("--full-cv", action="store_true",
                         help="全foldでCV（遅いが精度重視）")

    # ── predict ─────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="今日の予想を生成")
    p_pred.add_argument("--data",  required=True,
                        help="fan/B ファイルが置かれたディレクトリ")
    p_pred.add_argument("--date",  required=True,
                        help="開催日 (YYYY-MM-DD)")
    p_pred.add_argument("--wind-speed",  type=float, default=0.0,
                        help="風速 (m/s, デフォルト: 0)")
    p_pred.add_argument("--wind-dir",    default="",
                        help="風向き: 追/向/横 (デフォルト: なし)")
    p_pred.add_argument("--tide",        default="",
                        help="潮位: 干/満 (デフォルト: なし)")

    # ── result ──────────────────────────────────────────────
    p_res = sub.add_parser("result", help="結果照合")
    p_res.add_argument("--data",  required=True,
                       help="fan/B/K ファイルが置かれたディレクトリ")
    p_res.add_argument("--date",  required=True,
                       help="開催日 (YYYY-MM-DD)")
    p_res.add_argument("--wind-speed",  type=float, default=0.0)
    p_res.add_argument("--wind-dir",    default="")
    p_res.add_argument("--tide",        default="")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "config":
        cmd_config(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "result":
        cmd_result(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
