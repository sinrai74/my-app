# 上位監査記録：M15 — 正式条件外の buy_decision 差分が streak に与える影響（現行実装の事実確認）

作成日: 2026-10-06 ／ ステータス: **監査結果の記録（現行実装の事実の記録のみ・正式仕様の決定なし・既存記録の変更なし）**

---

## 1. 本書の位置付け

本書は、現行の Shadow 実装において、buy_decision の差分が streak の判定と Go/No-Go の入力にどう影響するかを、コードの読み取りにより確認した結果を記録する。

本書は現行実装の事実を記録するものであり、正式仕様を決定するものではない。

- 基準 commit：`558c2e7`（origin/main）
- 確認したファイル：`rebuild/shadow/runner.py`、`rebuild/shadow/legacy_values_builder.py`、`rebuild/shadow/aggregator.py`、`rebuild/shadow/staged_comparator.py`、`rebuild/actions/shadow_entrypoint.py`、`rebuild/shadow/go_no_go.py`
- 監査はコードと既存記録の読み取りのみで実施した。コード・docs・テストの変更、テストの実行、Shadow の実行、Gmail への送信、Release の操作、データの生成、commit、push は行っていない。

### 区分

- FACT：今回、コードから直接確認した事実
- EXISTING RECORD：以前から正式記録に存在する内容
- UNRESOLVED：今回の調査でも正式には決定していない内容

---

## 2. FACT

| # | 事実 | 箇所 |
|---|---|---|
| F1 | `ShadowRunner.run_and_compare` の比較ループで、buy_decision は段名 `"buy_decision"` として比較される。diff の `field_path` は `$.buy_decision.<field>` となる | `runner.py` 比較ループ（`("buy_decision", buy_decision)`、`compare(..., path=f"$.{name}")`） |
| F2 | buy_decision は、Legacy 側に `buy_decision`（sent の `purchase_view()`）が存在し、かつ Rebuild 側の buy_decision が None でない場合に比較される。`purchase_view()` が None の sent 行では、Legacy 側にキーがなく、比較されない | `runner.py`、`legacy_values_builder.py` `build_legacy_values` |
| F3 | `legacy_values_builder.py` は、Legacy 側に `buy_assessment` のキーを生成しない（生成するのは race・evaluation・buy_decision のみ）。そのため、buy_assessment は比較されていない | `legacy_values_builder.py`、`runner.py` |
| F4 | `STREAK_EXCLUDED_STAGES = {"evaluation"}` であり、buy_decision は除外対象ではない | `aggregator.py` |
| F5 | `$.buy_decision.*` の diff が1件でもあると、その diff は streak 判定用の diff に残り、`stopped_at = "diff"`、`all_matched = False` となる | `aggregator.py` `to_staged_result`、`staged_comparator.py` `StagedResult.all_matched` |
| F6 | その場合、`ConsecutiveMatchCounter.record()` の else 分岐に入り、そのレースの eval_id が `broken_at` に追加され、`current_streak = 0` となる | `staged_comparator.py` `ConsecutiveMatchCounter.record` |
| F7 | その場合、`max_streak` は更新されず、それまでの値が維持される | 同上 |
| F8 | Go/No-Go には、実行終了時点の `current_streak` が `shadow_consecutive_matches` として渡される。go_no_go は、これを「G2/G3: Shadow連続一致」の項目で `shadow_required_matches` と比較する | `shadow_entrypoint.py` `run_shadow_multiple`、`go_no_go.py` |
| F9 | したがって、現行実装では、buy_decision の差分は streak を 0 に戻し、Go/No-Go の「G2/G3」の項目の入力値に影響する | F1〜F8 |
| F10 | `runner.py` のコメントは、buy_decision を「Stage 2で追加するG3-B/G3-C比較用」として扱っている | `runner.py` |

---

## 3. EXISTING RECORD

| # | 内容 | 出典 |
|---|---|---|
| R1 | Stage 3 Run 36224354755 で、`$.buy_decision.n_bets` の差分は46件（legacy/rebuild が 1/0 で37件、2/0 で8件、2/3 で1件） | `docs/Stage3_Shadow観測実測記録.md` §3.3・§3.4 |
| R2 | 同 Run で、`$.buy_decision.purchased_amounts.__len__`・`$.buy_decision.purchased_combos.__len__` の差分は各46件、`$.buy_decision.purchased` の差分は0件 | 同 §3.3 |
| R3 | 同 Run の集計は、matched 1・diff 46・skipped 43・max_streak 1・broken_at 46件である | 同 §3 |
| R4 | 同 Run で、race の4キーの差分は0件である | 同 §3 |
| R5 | matched の1件 `20260704_09_06` は、evaluation 段の差分（`$.evaluation.race_type`）のみのレースであり、max_streak=1 の要因となっている | 同 §4.1 |
| R6 | 継続基準書 §8.5 は、「47比較レース中46件が buy_decision差も併発しており、buy_decision差は従来どおり streak をリセットする」と明記している | 継続基準書 §8.5 |
| R7 | §8.6 は、buy_decision が Stage 2（`d974f0f`）で追加された比較・観測項目であり、Step6-0 §⑤ 等の正式受入条件に採用された記録はないとしている | 継続基準書 §8.6 [A] |
| R8 | §8.6 は、[B] の方針として、buy_decision を正式受入条件へ昇格させず、streak から除外することも決定せず、「正式条件外の追加観測項目」として扱うとしている | 継続基準書 §8.6 [B] |
| R9 | §8.6 は、「正式条件外の項目が現在のstreakに影響している」不整合を、既知の未解決事項としている | 継続基準書 §8.6 |
| R10 | §8.6 は、「現在のNO_GOは正式条件外の buy_decision 差が実装上発生させているものであり、これをそのまま正式G2/G3のNO_GOとは扱えない」としている | 継続基準書 §8.6 |
| R11 | `docs/上位判断記録_G2_G3_Phase1_20260927.md` は、§8.6 の buy_decision の扱いを [B] として確認し、上記の不整合の解消方針は定めないとしている | 同記録 §3.2・§3.4 |
| R12 | `docs/上位監査記録_G2_G3状態整理_20261006.md` は、buy_decision の差分の streak への影響を USER DECISION の論点として整理している | 同記録 §3・§8 |

**既存記録上の位置付け**：既存記録では、「buy_decision の差分が streak をリセットしている」という現象は認識されている（R6・R9・R10）。しかし、それを正式仕様として承認した記録はない。また、streak から除外する判断もされていない（R8）。

**記録と実装の対応**：R3・R4 により、同 Run では evaluation 段以外の差分は buy_decision 段の差分のみであった。このため、broken_at の46件は、F5・F6 の経路により buy_decision の差分で生じたものと整合する。これは、記録の数値と実装の経路を組み合わせて得られる対応であり、新たな実行による確認ではない。

---

## 4. UNRESOLVED

| # | 未確定事項 |
|---|---|
| U1 | buy_decision の差分を、正式な streak の判定に含めるか、除外するか |
| U2 | buy_decision（G3-B・G3-C）を、Step6-0 §⑤ の正式G3 に含めるか。G3-3（§8.6 と §9 の優先関係）および C8 の位置付けの未確定と関係する |
| U3 | 正式G3（BuyAssessment）が現行では比較されていない（F3）状態で、Go/No-Go の「G2/G3」の項目に何を入力すべきか |
| U4 | Go/No-Go の「G2/G3」の項目が、現行実装上、1つの streak で判定されていること（F8）を、正式にどう位置付けるか |
| U5 | buy_decision の差分（Legacy で購入、Rebuild で見送り）の原因。既存の原因調査の保留（FROZEN）を維持する（`docs/Stage3_Shadow観測_原因調査記録.md`・同追補） |
| U6 | broken・streak の対象とする段の正式な範囲。`docs/上位監査記録_M8_M9_M10_現行実装事実確認_20261006.md` の U8（broken の対象とする段）・U9（buy_decision を broken・streak の判定に含めることの正式性）と同じ論点である |

---

## 5. 本書の境界

- 本書は現行実装の事実を記録するものであり、正式仕様を決定しない。
- F9 から、「buy_decision を正式な streak の対象にすべき」とは導かない。
- 同様に、「buy_decision を streak の対象外にすべき」とも決定しない。
- G2・G3 の正式条件を変更しない。
- buy_decision を正式G3 へ昇格しない。
- buy_decision を streak から除外しない。
- G2改を採用しない。
- G3-3（§8.6 と §9）の優先順位を変更しない。
- C8 の位置付けを決定しない。
- Stage 3 の完了を判定しない。
- Stage 6 の開始を判定しない。

---

## 6. 既存記録との関係

- 本書は、継続基準書（§8.5・§8.6 を含む）、Phase0.5 設計固定書 v1.1.8、Step5-0、Step6-0、`docs/上位判断記録_G2_G3_Phase1_20260927.md`、`docs/Stage3_Shadow観測実測記録.md`、`docs/Stage3_Shadow観測_原因調査記録.md`（同追補を含む）、`docs/上位監査記録_G2_G3状態整理_20261006.md`、G3-3 関連の3記録、`docs/上位監査記録_M8_M9_M10_現行実装事実確認_20261006.md`、その他の上位判断記録・上位監査記録の各件を改訂・置換しない。
- 新しい完了条件、判定基準、Go/No-Go 条件、工程番号は定義しない。
