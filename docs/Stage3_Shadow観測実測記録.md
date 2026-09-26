# Stage3_Shadow観測実測記録

作成日: 2026-09-26 ／ ステータス: **観測記録（実測値の記録のみ・判定なし）**

---

## §0 本書の位置づけ

- 継続基準書 `docs/再設計_完成基準・残作業・再開手順書.md` §12 NEXT ACTION「Stage 3：Shadow測定」
  に基づき、既存の Shadow 実装が出力した測定値を記録する。
- 本書は**観測方式B（正式受入判定とは切り離し、実測・記録のみ）**によるものであり、
  合否判断・設計判断・新基準の制定を目的としない。
- 既存実装（`shadow_run.yml` / `shadow_entrypoint.py` / comparator / runner / Legacy / Rebuild）は
  変更していない。
- 本書は Phase0.5 §⑥ の保存対象表にない新規文書であり、§⑥ L439 に基づきレビュー済みの
  新規文書追加として作成する。**保存対象表自体は変更しない。**

### 本書で判断しないこと

- Stage 3 の完了
- Go/No-Go の確定
- Phase 1 開始条件（Phase0.5 §⑱）および No.14
- C8 / G2改 / 正式G2 / 正式G3 / G3-B・G3-C の定義および正式受入条件化
- 100レース連続の判定ルール、skip / broken の解釈

---

## §1 実行条件

| 項目 | 値 |
|---|---|
| Run ID | 36224354755 |
| Run URL | https://github.com/sinrai74/my-app/actions/runs/36224354755 |
| Workflow | shadow-run（`.github/workflows/shadow_run.yml`。変更なし） |
| 実行日時 | 2026-09-26 06:38:19Z 〜 07:09:07Z（約31分） |
| commit SHA | b13658dede0f2b4d49b4360780b09a6495c07eb9 |
| target_races | 修正版 **90件** |
| target_races と sent_20260704.txt | **完全一致・重複なし**（`parse_target_races()` と正規化キー集合で検証済み） |
| sent 読み込み | `sent_20260704.txt` count=**90** |
| 固定条件 | USE_REBUILD_PIPELINE=False ／ SHADOW_MODE=1 |
| 成果物 | `rebuild/shadow_diff_report.json`（Actions artifact `shadow-diff-report`） |

---

## §2 事前条件（Golden回帰）

`golden_100_percent` の入力値は **1**。その根拠は以下の実行結果である。

| 項目 | 値 |
|---|---|
| 対象 commit | b13658dede0f2b4d49b4360780b09a6495c07eb9 |
| 実行環境 | Linux |
| コマンド | `cd rebuild && python -m pytest tests/regression -q` |
| 結果 | **25 passed / 0 failed / 0 errors / 0 skipped / 670 subtests passed** |

---

## §3 観測結果（Shadow が出力した値をそのまま記録）

### 3.1 集計値

| 項目 | 値 |
|---|---|
| total_races | 90 |
| matched_races | 1 |
| diff_races | 46 |
| skipped_races | 43 |
| shadow_consecutive_matches | 0 |
| max_streak | 1 |
| broken_at | 46件 |
| required | 100 |
| satisfied | false |

### 3.2 skip の内訳

| 理由 | 件数 |
|---|---|
| `_evaluate_bets returned an empty list; no bet candidate is available for this race (no default value is supplied)` | 43 |
| `legacy value not found in sent records` | **0** |

skip 43件の reason はユニークで1種類のみであった。

### 3.3 差分フィールド別件数

| field_path | 件数 |
|---|---|
| `$.evaluation.race_type` | 47 |
| `$.buy_decision.n_bets` | 46 |
| `$.buy_decision.purchased_amounts.__len__` | 46 |
| `$.buy_decision.purchased_combos.__len__` | 46 |
| `$.evaluation.upset_score` | 29 |
| `$.buy_decision.cost` | 1 |
| `$.buy_decision.purchased_amounts[0]` | 1 |
| `$.buy_decision.purchased_amounts[1]` | 1 |
| `$.buy_decision.purchased_combos[0]` | 1 |
| `$.buy_decision.purchased_combos[1]` | 1 |
| `$.evaluation.danger_score` | 0 |
| `$.buy_decision.purchased` | 0 |
| `$.race.*`（race 4-key） | 0 |

### 3.4 差分値の内訳

**`$.buy_decision.n_bets`（46件）**

| legacy / rebuild | 件数 |
|---|---|
| 1 / 0 | 37 |
| 2 / 0 | 8 |
| 2 / 3 | 1 |

**`$.evaluation.upset_score`（29件）**

| 項目 | 値 |
|---|---|
| 差の絶対値 | 0.0094 〜 6.274 |
| 最大差のレース | `20260704_21_06`（Legacy=6.274 / Rebuild=0.0） |

### 3.5 自動出力された Go/No-Go 情報の扱い

`shadow_diff_report.json` には `satisfied` / `go_no_go_decision` / `go_no_go_reasons` /
`go_no_go_pendings` が含まれるが、これは実装（`go_no_go.py` ほか）による自動出力である。
**本書ではこれらを正式な Go/No-Go 判断として解釈しない。**

---

## §4 特異点（観測された事実のみ）

### 4.1 `20260704_09_06`

`diff_count=1`（`$.evaluation.race_type` の差分1件）であるが、集計上は matched として計上され、
max_streak=1 の要因となっている。

### 4.2 `20260704_04_04`（diff_count=9）

artifact の実値は以下のとおり。

| field_path | legacy | rebuild |
|---|---|---|
| `$.evaluation.race_type` | `1残り荒れ型` | （空文字） |
| `$.buy_decision.cost` | 0 | 900 |
| `$.buy_decision.n_bets` | 2 | 3 |
| `$.buy_decision.purchased_amounts.__len__` | 2 | 3 |
| `$.buy_decision.purchased_amounts[0]` | 0 | 300 |
| `$.buy_decision.purchased_amounts[1]` | 0 | 300 |
| `$.buy_decision.purchased_combos.__len__` | 2 | 3 |
| `$.buy_decision.purchased_combos[0]` | `1-6-3` | `3-5-6` |
| `$.buy_decision.purchased_combos[1]` | `1-2-6` | `1-3-5` |

本Run中、要素値（`[0]` / `[1]`）の差分が記録された唯一のレースである。

---

## §5 前回Run（36222680953）との関係

| 項目 | 前回 36222680953 | 今回 36224354755 |
|---|---|---|
| target_races | 90件（うち **33件が sent に存在しない誤指定**） | 修正版90件（sent と完全一致） |
| legacy value not found | 33件 | **0件** |
| total / matched / diff / skipped | 90 / 0 / 31 / 59 | 90 / 1 / 46 / 43 |

前回Runは target_races の指定誤りにより、sent に存在しない33件を含んでいた。
本書では **今回Run（36224354755）を Stage 3 の修正版観測結果として扱う**。

---

## §6 未調査事項（本書では原因を推測・断定しない）

1. `20260704_09_06` が `diff_count=1` でありながら matched として計上される仕組み
2. `20260704_04_04` で Rebuild 側のみ購入が発生した理由
3. `n_bets` が Rebuild 側で 0 となる内部理由
4. `upset_score` の差の最大値が 6.274 となった理由

---

## §7 本書で追加していないもの

- 新しい合格基準・受入条件
- C8 の定義、および継続基準書が参照する13章相当の内容
- G2改の定義
- G3-B / G3-C の正式受入条件化
- 100レース連続の判定ルール
- skip / broken の新しい解釈
- 一致率・達成率等の算出

観測方式B（実測・記録のみ）を維持している。

---

## §8 変更していないもの

- `shadow_run.yml` / `shadow_entrypoint.py` / comparator / runner
- Legacy コード / Rebuild 実装
- `docs/Phase0_5_設計固定書.md`（§⑥ の保存対象表を含む）
- `docs/Phase1開始条件_正式判定記録.md`
- Go/No-Go 基準 / Feature Freeze 対象
