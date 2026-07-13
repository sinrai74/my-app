# インシデント記録: Freeze宣言前の未コミット実装の履歴整理

記録日: 2026-07-13
関連: 設計書 Phase0.5 v1.1 ⑳20.3（凍結期間中の例外規定）

## 経緯

Feature Freeze宣言（2026-07-13、タグ `freeze-v1-baseline`）に向けた Step 0 作業中、
`git status` にて以下がstage済み・未コミットのまま残っていることが判明した。

- `x_results_common.py`（修正）
- `x_results_public.py`（修正）
- `files.zip`（削除）
- `daily_stats.json`（Actions側と手元側の競合）
- 40件超のuntrackedファイル

確認の結果、`x_results_common.py` / `x_results_public.py` の変更は
「【Phase2】万舟の平均払戻・最大払戻」を `calc_brand_results` に追加する実装であり、
**Feature Freeze宣言よりも前の時点でローカルに存在していた作業**であることを確認した。

## 対応

1. Freeze宣言（Step 0）のコミットには、この変更を含めず、
   `x_ranking.yml`（tuner自動反映停止）と `docs/freeze_baseline.md`（凍結記録）のみで確定した
   （コミット `d258757`、タグ `freeze-v1-baseline`）
2. `daily_stats.json` の競合は Updated upstream（GitHub Actions側）を採用し、
   手元のStashed側の変更は破棄した
3. `files.zip` の削除、および40件超のuntrackedファイルは今回のコミット対象から除外し、
   別途整理する（本記録では未対応のまま保留）
4. Phase2変更（x_results_common.py / x_results_public.py）は、
   「新規変更」ではなく「凍結前に存在していた実装を履歴として確定する」作業として、
   Step 0とは別コミットで記録した

## 位置づけ

本コミットは設計書⑳20.2（新機能追加・アルゴリズム変更の禁止）に対する例外ではなく、
**凍結宣言前の既存状態を確定させたもの**である。したがって⑳20.3の事後記録要件に基づき、
本ファイルを記録として残す。

この変更内容自体（万舟の平均払戻・最大払戻の追加）が今後の比較基準
（freeze-v1-baseline）に含まれることになるため、⑭のゴールデンデータ・回帰テストは
本コミットを含んだ状態を「正」として扱う。

## 未対応事項（本記録の対象外・別途整理）

- files.zip の削除の要否
- untrackedファイル40件超の分類・扱い（一覧取得を別途実施）
