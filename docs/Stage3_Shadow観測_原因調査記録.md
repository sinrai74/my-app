# Stage3_Shadow観測_原因調査記録

作成日: 2026-09-26 ／ ステータス: **原因調査記録（既存証跡とコードの照合結果の記録のみ・判定なし）**

---

## §0 本書の位置づけ

- `docs/Stage3_Shadow観測実測記録.md`（Run 36224354755 の実測値の正式記録）§6 に
  「未調査事項」として列挙した4点について、その後に実施した原因調査の結果を記録する。
- 本書は既存の実測記録とは**別の文書**である。`docs/Stage3_Shadow観測実測記録.md` は変更しない。
- 本書は**観測方式B（正式受入判定とは切り離し、実測・記録のみ）**を維持し、
  合否判断・設計判断・新基準の制定・次工程の決定を目的としない。
- 本書は Phase0.5 §⑥ の保存対象表にない新規文書であり、**§⑥ L439 のレビュー対象
  （本書作成時点で未実施）**。保存対象表自体は変更しない。

### 本書で判断しないこと

- Stage 3 の完了・未完了
- Go/No-Go の確定、受入判定
- Phase 1 開始条件（Phase0.5 §⑱）および No.14
- C8 / G2改 / 正式G2 / 正式G3 / G3-B・G3-C の定義および正式受入条件化
- 100レース連続の判定ルール、skip / broken の解釈
- 各調査結果についての是非・評価
- 今後の扱い（次工程）

---

## §1 調査対象・基準Run

| 項目 | 値 |
|---|---|
| 基準Run | 36224354755（`docs/Stage3_Shadow観測実測記録.md` §1） |
| Run 実行時 commit | b13658dede0f2b4d49b4360780b09a6495c07eb9 |
| 照合したコードの commit | `b13658d`（本書作成時点の HEAD `85d343b` との差分は `docs/Stage3_Shadow観測実測記録.md` の追加1件のみで、`rebuild/` 配下に差分なし） |
| 調査対象 | 実測記録 §6 の4点（下表） |

| # | 実測記録 §6 の記載 |
|---|---|
| ① | `20260704_09_06` が `diff_count=1` でありながら matched として計上される仕組み |
| ② | `20260704_04_04` で Rebuild 側のみ購入が発生した理由 |
| ③ | `n_bets` が Rebuild 側で 0 となる内部理由 |
| ④ | `upset_score` の差の最大値が 6.274 となった理由（`20260704_21_06`：Legacy=6.274 / Rebuild=0.0） |

---

## §2 調査方法・参照した既存証跡

### 2.1 方法

- 既存の証跡（実測記録・継続基準書）と、`b13658d` 時点のコードの読み取りによる照合のみを行った。
- Shadow の再実行、データの再生成、中間値の再計算は行っていない。

### 2.2 参照した証跡

| 種別 | 対象 |
|---|---|
| 実測記録 | `docs/Stage3_Shadow観測実測記録.md`（§3.3・§3.4・§4.1・§4.2・§6） |
| 継続基準書 | `docs/再設計_完成基準・残作業・再開手順書.md` §8.5 |
| commit | `3b98a8e`（feat: evaluation段をGo/No-Go streak必須判定から除外（観測は継続）） |
| コード | `rebuild/shadow/aggregator.py` |
| コード | `rebuild/shadow/prediction_provider.py`（`_EvaluateBetsCapture`） |
| コード | `rebuild/pipelines/buy_decision_builder.py`（`DefaultBuyDecisionBuilder`） |
| コード | `rebuild/actions/shadow_entrypoint.py`（`build_bundle`・複数レース実行ループ） |
| コード | `rebuild/pipelines/evaluation_pipeline.py`（`persist` の扱い） |
| コード | `rebuild/core/upset.py`（`calculate_upset_score`） |

行番号はいずれも `b13658d` 時点のものである。

---

## §3 調査結果

### 3.1 ① `20260704_09_06` の matched 扱い　【原因確定】

**実測値**（実測記録 §4.1）：`diff_count=1`。差分は `$.evaluation.race_type` の1件のみ。

**実装上の事実**

| 箇所 | 内容 |
|---|---|
| `shadow/aggregator.py` L53 | `STREAK_EXCLUDED_STAGES = frozenset({"evaluation"})` |
| 同 L56〜66 `_is_streak_excluded` | `field_path` の先頭の段名（`$.<stage>.…` の `<stage>`）が `evaluation` であれば streak 判定対象外とする |
| 同 L69〜92 `to_staged_result` | streak 判定用の diff から evaluation 段を除外し、残りが空なら `stopped_at=None`（`all_matched=True`）とする。`diffs` には evaluation 段を含む全件を残す |
| 同 L136〜149 `record` | `all_matched` が真なら matched、偽なら diff として計上する |
| 継続基準書 §8.5 | evaluation 段（upset_score / race_type）は観測・比較・diff記録を継続するが、streak の必須判定対象からは外す方針（承認済み）。実装は commit `3b98a8e` |

**経路**：`20260704_09_06` の唯一の差分 `$.evaluation.race_type` は evaluation 段に属する
→ streak 判定用の diff が空になる → `all_matched=True` → matched として計上される。
一方、`shadow_diff_report.json` の `diffs` には当該差分が残るため、`diff_count=1` と記録される。

### 3.2 ② `20260704_04_04` の購入差　【原因経路確定】

**実測値**（実測記録 §4.2）：Legacy は `n_bets=2`・combo `1-6-3` / `1-2-6`・金額 0 / 0・`cost=0`、
Rebuild は `n_bets=3`・combo `3-5-6` / `1-3-5` / …・金額 300 / 300 / …・`cost=900`。

**比較される値の出所**

| 側 | 値の出所 |
|---|---|
| Legacy 側 | `sent_20260704.txt` に保存された当日の記録値 |
| Rebuild 側 | Shadow 実行時に Legacy の `_evaluate_bets` を再実行した結果を、`DefaultBuyDecisionBuilder` が束ねた値 |

**実装上の事実**

| 箇所 | 内容 |
|---|---|
| `actions/shadow_entrypoint.py` L229・L236〜255 `build_bundle` | `_EvaluateBetsCapture` を生成し、`prediction_provider` の `evaluate_bets` と `purchase_result_source` の両方に同一インスタンスを渡す。`decision_builder=DefaultBuyDecisionBuilder()` |
| `shadow/prediction_provider.py` L162〜234 `_EvaluateBetsCapture` | Legacy `notify_arashi._evaluate_bets` を1回呼び、戻り値のリスト全体を保持する。`last_purchase_result(eval_id)` はそのリストを `LegacyPurchaseResult.purchases` として返す |
| `pipelines/buy_decision_builder.py` L58〜 `DefaultBuyDecisionBuilder` | `purchases` の先頭要素の `purchased` で購入/見送りを分岐し、購入時は各要素の `combo` と `amount` をそのまま `purchased_combos` / `purchased_amounts` とし、`n_bets=len(candidates)`、`cost=sum(amounts)` とする（L88〜118） |

**経路**：Shadow 比較における Rebuild 側の購入対象は、Rebuild 独自の買い目生成ロジックによるものではなく、
**Shadow 実行時に再実行された Legacy `_evaluate_bets` の戻り値**が `DefaultBuyDecisionBuilder` を通って出力されたものである。
`20260704_04_04` の差は、sent の当日記録値と、Shadow 実行時の `_evaluate_bets` 再実行結果との差として現れている。

**本書で特定していないこと**：Shadow 実行時の `_evaluate_bets` が当日と異なる戻り値を返した要因
（入力・外部状態のどれが異なったか）。Shadow 実行時の入力値は保存されていない（§4.1）。当日の入力値が sent に
どこまで記録されているかは、本調査では確認していない。

### 3.3 ③ `n_bets=0` の45件　【コード経路確定・個別の判定理由は未取得】

**実測値**（実測記録 §3.4）：`$.buy_decision.n_bets` の差分46件のうち、Rebuild 側が 0 のものは
Legacy=1 / Rebuild=0 が37件、Legacy=2 / Rebuild=0 が8件の計45件。

**実装上の事実**

| 箇所 | 内容 |
|---|---|
| `pipelines/buy_decision_builder.py` L73〜85 | `purchases` が空のとき `n_bets=0` を返す分岐 |
| 同 L88〜101 | 先頭要素の `purchased` が偽（見送り）のとき `n_bets=0` を返す分岐 |
| `shadow/prediction_provider.py` L209〜214 | `_evaluate_bets` の戻り値が空リストの場合、`ValueError("_evaluate_bets returned an empty list; …")` を送出する |
| `actions/shadow_entrypoint.py` L437〜448 | 上記メッセージの `ValueError` は skip（`skipped: True`）として記録し、比較せずに次のレースへ進む |

**経路**：Shadow の経路では、`_evaluate_bets` が空リストを返したレースは builder に到達する前に skip となる
（実測記録 §3.2 の skip 43件）。したがって、比較結果に `n_bets=0` として現れる45件は、コード経路上
**見送り（`purchased=False`）の分岐**によるものである。

**本書で特定していないこと**：45件それぞれについて、Shadow 実行時の `_evaluate_bets` がなぜ見送りと判定したか
という個別の判定理由。今回の調査ではこれを取得していない。

### 3.4 ④ `20260704_21_06` の `upset_score=0.0`　【原因未確定】

**実測値**（実測記録 §3.4）：Legacy=6.274 / Rebuild=0.0（`upset_score` 差分29件中の最大差）。

**実装上の事実**：`b13658d` 時点の `core/upset.py` `calculate_upset_score` において、
最終値 `upset_score` が `0.0` になり得る経路として、今回のコード照合で以下の4経路を確認した。

| # | 箇所 | 内容 |
|---|---|---|
| a | L224〜226 | 安心レースのゼロ化：`boat1_prob > 0.65` かつ `best_other_prob < boat1_prob` のとき `upset_score = 0.0` |
| b | L230〜232 | 1号艇A1：1号艇の `racer_class == "A1"` のとき `upset_score = 0.0` |
| c | L233〜235 | 1号艇A2：1号艇の `racer_class == "A2"` のとき `upset_score = max(upset_score - 1.5, 0.0)`（直前の `upset_score` が 1.5 以下なら `0.0`） |
| d | L216〜217 | `upset_prob` の下限クランプ：`upset_prob = max(0.0, min(upset_prob, 0.95))`、`upset_score = upset_prob * 10.0`（`upset_prob` が 0 以下なら `0.0`） |

- a と b は独立した `if` として順に評価されるため、**同時に成立し得る**。
- b と c は `if` / `elif` の関係にあり、同時には適用されない。
- 調査過程の整理では a・b の2条件としていたが、本書では上記コード照合の結果に合わせ、4経路として記録する。

**実測への適用**：コード上の到達経路としては、`calculate_upset_score` 内の a〜d を確認した
（同関数の戻り値は `core/engine.py` L135・L241・L291 で RaceEvaluation の `upset_score` にそのまま渡される）。
今回の実測値 0.0 に対する適用結果は、証跡がないため特定できない。

| 必要な証跡 | 状態 |
|---|---|
| `boat1_prob`・`best_other_prob`・`upset_prob` 等の中間値 | 保存されていない |
| 1号艇の `racer_class` 等、計算時の入力値 | 保存されていない |
| 各分岐の適用結果（`grade_filter_note` を含む `UpsetResult` の内容） | 保存されていない |
| 当該レースの RaceEvaluation | 保存されていない（§4.1） |

したがって、**④は原因未確定**である。本書では a〜d のいずれかを適用された経路として特定しない。
また、Legacy 側の値 6.274 と Rebuild 側の値 0.0 の差が生じた要因についても特定しない。

---

## §4 観測証跡上の制約（今回の調査で確認された事実）

### 4.1 Shadow 実行で残る証跡と残らない証跡

| 箇所 | 事実 |
|---|---|
| `actions/shadow_entrypoint.py` L244 | Shadow の `build_bundle` は `durable_store=None` で構成される |
| `pipelines/evaluation_pipeline.py` L99〜135 | RaceEvaluation の保存は `persist=True` かつ `durable_store` が注入された場合のみ行われる |
| `actions/shadow_entrypoint.py` L452〜456 | 比較したレースについてレポートへ出力されるのは `eval_id`・`diff_count`・`diffs` のみ |
| 同 L443〜447 | skip したレースについて出力されるのは `eval_id`・`skipped`・`reason` のみ |

### 4.2 これにより確認された制約

- Shadow の差分レポートには、差分のあったフィールドの Legacy 値・Rebuild 値は残るが、
  RaceEvaluation、upset 計算の中間値、分岐条件の適用結果、`_evaluate_bets` の判定に至る中間状態、
  計算時の入力値は残らない。
- そのため、過去Runについて、差分値から原因を完全に逆算できない場合がある。今回の④がこれに該当し、
  ②で当日と Shadow 実行時の結果が異なった要因、③の個別の判定理由も、同じ理由で取得していない。

本節は事実の記録であり、観測基盤・S6（除外理由record を含む）・RaceEvaluation の保存仕様・ログ出力について、
変更・追加の要否を判断するものではない。

---

## §5 原因確定／原因未確定の区別

| # | 対象 | 区分 | 確定している範囲 | 確定していない範囲 |
|---|---|---|---|---|
| ① | `20260704_09_06` matched | **原因確定** | evaluation 段の diff が streak 判定から除外される経路（§3.1） | ― |
| ② | `20260704_04_04` 購入差 | **原因経路確定** | Rebuild 側の値が Shadow 実行時の Legacy `_evaluate_bets` 再実行結果である経路（§3.2） | Shadow 実行時の戻り値が当日と異なった要因 |
| ③ | `n_bets=0` 45件 | **コード経路確定** | 見送り（`purchased=False`）分岐によること（§3.3） | 45件それぞれの個別の判定理由（未取得） |
| ④ | `20260704_21_06` upset 0.0 | **原因未確定** | `calculate_upset_score` の最終値が 0.0 になり得るコード上の経路が a〜d であること（§3.4） | 実測でどの経路が適用されたかは特定できない（証跡なし） |

---

## §6 本書で実施していないこと

- Shadow の再実行
- データの再生成、中間値の再計算
- コード変更（Legacy・Rebuild・comparator・runner・aggregator・test・config を含む）
- 設計変更（Phase0.5 設計固定書・継続基準書・S6 を含む）
- 既存文書の変更（`docs/Stage3_Shadow観測実測記録.md` を含む）
- 各調査結果の是非の評価
- 新しい完了条件・判定基準・例外ルールの設定
- 今後の扱い（次工程）の決定

---

## §7 調査終了時点の状態

| 項目 | 状態 |
|---|---|
| 実測記録 §6 の4点 | 本書 §3・§5 のとおり記録した（① 原因確定、② 原因経路確定、③ コード経路確定・個別理由未取得、④ 原因未確定） |
| 基準Run の実測値 | `docs/Stage3_Shadow観測実測記録.md` の記載から変更なし |
| 観測方式 | 観測方式B を維持 |
| Stage 3 の完了・未完了、Go/No-Go | 本書では判断していない |
| 本書の §⑥ L439 レビュー | 未実施 |

本書の記録をどのように扱うかは、**判断事項として残す**。本書では定めない。
