# 上位判断記録：「100連続」・skip・broken — D4 のうち、不一致時の current_streak の扱い

作成日: 2026-10-07 ／ ステータス: **上位判断の記録（判断内容と根拠の記録のみ・実装なし・既存記録の変更なし）**

---

## 1. 本書の位置付け

本書は、作業上の streak 計算・観測において、比較可能なレースが不一致（diff）となった場合の current_streak の扱いについて、ユーザーの上位判断を記録する。

P1P2記録 §1.1 は、D4 を「current_streak・max_streak の扱い、および不一致と No-Go の関係」としている。本書が判断するのは、**D4 のうち、不一致時の current_streak の扱いだけ**である。D4 のその他の部分（max_streak の役割、判定値、不一致と No-Go の関係）は判断しない。本書によって D4 全体が判断されたものではない。

本書は、次の判断記録を前提とする。

- `docs/上位判断記録_100連続_P1P2適用範囲_20261007.md`（以下「P1P2記録」）
- `docs/上位判断記録_100連続_D1_1件定義_20261007.md`（以下「D1記録」）
- `docs/上位判断記録_100連続_D2_skip扱い_20261007.md`（以下「D2記録」）

本書は、P1 により整理の対象とした継続基準書 §8.5 [B] の作業上の streak 計算・観測方法についての判断である。

- 基準 commit：`b0d3393`（origin/main）
- 本書の作成にあたり、コード・既存 docs の変更、テストの実行、Shadow の実行、Gmail への送信、Release の操作、データの生成は行っていない。
- 過去のチャットの記録は、根拠として用いていない。

### 1.1 用語

| 用語 | 本書での意味 |
|---|---|
| 比較可能なレース | D2 のレース単位の skip に当たらず、比較が行われた eval_id（D1） |
| 不一致（diff） | 比較可能なレースが、作業上の一致判定で一致とならなかったこと。何をもって一致・不一致とするか（一致判定の内容）は、本書では定めない |

---

## 2. FACT（既存資料・コード・artifact から直接確認できる事実）

| # | 事実 | 出典 |
|---|---|---|
| F1 | 正式受入条件 G2/G3 は「Shadow並走でRaceEvaluation・BuyAssessmentの差分が100レース連続ゼロ」と記載されている | Step6-0 Shadow運用計画・実装前レビュー §⑤「Go判定の手順」2 |
| F2 | Step5-0 は、G2 を「100レース連続でRaceEvaluation差分ゼロ（1件でも不一致ならNo-Go）」と記載している。Step5-0 §⑥ は、ユーザーに承認された項目に含まれていない | Step5-0 境界確定設計書 §⑥、`docs/上位判断記録_Phase1前提_hit_record_共通ファイル_20261002.md` §2.4 |
| F3 | 不一致時の current_streak の扱いを正式に定めた既存資料はない。「1件でも不一致ならNo-Go」と streak のリセットを同一とした正式な記録もない（M8-U5） | `docs/上位監査記録_M8_M9_M10_正式定義突合_20261006.md`（以下「M8〜M10正式定義突合」）§2・§2.3・§6 |
| F4 | 現行実装の `ConsecutiveMatchCounter.record()` は、一致なら current_streak を +1 して max_streak を更新し、不一致なら broken_at に eval_id を追加したうえで current_streak を 0 にする。current_streak の更新は `all_matched` だけで決まり、broken_at の内容を参照しない | `rebuild/shadow/staged_comparator.py` `ConsecutiveMatchCounter.record` |
| F5 | 同クラスの docstring は、「途中で1件でも差分が出たら0から数え直し（100/105ではなく100連続）」と記載し、出典として「Step6-1レビュー(d)」を挙げている。その原資料は確認できない | `rebuild/shadow/staged_comparator.py` `ConsecutiveMatchCounter` docstring、`docs/上位監査記録_Step6-1レビュー_d_参照元確認_20261006.md` §7 |
| F6 | 既存のテストは、不一致で current_streak が 0 になること、および「105件中100一致でも、途中で差分があれば連続ではない」ことを検証している | `rebuild/tests/unit/test_shadow_step6.py` `test_diff_resets_streak`・`test_100_consecutive_required_not_100_of_105`・`test_nonempty_diffs_resets_streak`、`rebuild/tests/unit/test_evaluation_streak_exclusion.py` |
| F7 | current_streak は、satisfied の判定と GoNoGoCriteria の `shadow_consecutive_matches` に用いられ、broken_at は集計の表示と実行結果の出力にだけ用いられる。両者の用途は分かれている | `rebuild/shadow/staged_comparator.py`、`rebuild/actions/shadow_entrypoint.py` `run_shadow_multiple`、`rebuild/shadow/go_no_go.py` |
| F8 | Step6-3-69 §2.1 は、`shadow_consecutive_matches` の供給元を current_streak として「確定」としている。これは入力の供給元の確定であり、判定値の定義ではない | Step6-3-69 §2.1、M8〜M10正式定義突合 §2 |
| F9 | 継続基準書 §8.5 [B・承認済み] により、evaluation 段だけの差分は、作業上の streak の判定において不一致として扱われない | 継続基準書 §8.5、`rebuild/shadow/aggregator.py` `STREAK_EXCLUDED_STAGES` |

---

## 3. 判断内容

### 判断1

**作業上の streak 計算・観測において、比較可能なレースが不一致（diff）となった場合、current_streak を 0 に戻す。その後に比較可能なレースが一致した場合は、0 から数え直す。**

### 判断2：適用範囲

1. 本判断は、継続基準書 §8.5 [B] の作業上の streak 計算・観測方法（P1）における、不一致時の current_streak の扱いだけを定める。
2. 何をもって不一致とするか（一致判定の内容、比較段の範囲、段単位の未比較）は定めない。§8.5 の evaluation 段の除外（F9）を含め、一致判定に関する既存の扱いを変更しない。
3. **「不一致時に current_streak を 0 に戻すこと」と「broken・broken_at の意味」は、別の論点である。** 本判断は、broken の意味、broken_at に何を記録するかを定めない。
4. **「不一致時に current_streak を 0 に戻すこと」と「No-Go となること」は、同一の概念ではない。** 本判断は、不一致と No-Go の関係を定めない。
5. 本判断は、max_streak の役割、current_streak と max_streak のどちらを判定値とするかを定めない。

### 判断3：D2 との関係

D2 の判断（レース単位の skip は streak を増加もリセットもせず、次の比較可能なレースから継続する）は変更しない。本判断の対象は skip ではなく、比較可能なレースの不一致である。

D2 と本判断を合わせた作業上の扱いは、次のとおりである。

| 比較の結果 | current_streak の扱い | 根拠 |
|---|---|---|
| 比較可能なレースが一致 | +1 | 現行の扱い（本書では判断していない） |
| レース単位の skip | 維持（増加もリセットもしない） | D2記録 判断1 |
| 比較可能なレースが不一致 | 0 に戻す | 本書 判断1 |
| 不一致の後の一致 | 0 から数え直す | 本書 判断1 |

上表の「一致時に +1」は、現行の扱いを記載したものであり、本書で正式に判断したものではない。

---

## 4. 判断理由

1. 正式条件の文言は「100レース連続ゼロ」であり、差分ゼロのレースが「連続」することを求める表現になっている（F1）。不一致となったレースで連続が途切れ、数え直すという扱いは、この文言と整合する。ただし、本判断は正式G2・G3 の条件を解釈・再定義するものではなく、作業上の streak 計算についての判断である（P1）。
2. 不一致時の current_streak の扱いを正式に定めた既存資料はなく（F3）、本判断はその点について作業上の扱いを定める。
3. current_streak の更新は broken_at に依存せず、両者の用途も分かれている（F4・F7）。このため、本判断は broken の意味と切り離して判断できる。
4. 現行実装・docstring・テストも、不一致時に 0 へ戻す扱いになっている（F4〜F6）。ただし、これらは参考資料であり、本判断はそれらを正式仕様として採用したものではない（P2）。

---

## 5. FORMAL / REFERENCE / UNRESOLVED / CONTRADICTION

### 5.1 FORMAL

| 項目 | 内容 | 出典 |
|---|---|---|
| P1 の適用範囲 | 整理の対象は、まず継続基準書 §8.5 [B] の作業上の streak 計算・観測方法とする | P1P2記録 判断1 |
| D1 | 「1件」の粒度は eval_id（＝対象レース）とする | D1記録 判断1 |
| D2 | レース単位の skip は streak の判定対象外とし、増加もリセットもせず、次の比較可能なレースから継続する | D2記録 判断1 |
| 本判断 | 比較可能なレースが不一致となった場合、current_streak を 0 に戻し、その後は 0 から数え直す | 本書 判断1 |

### 5.2 REFERENCE（P2 により参照した判断材料。正式仕様としては扱わない）

| 資料 | 扱い |
|---|---|
| 現行実装（`ConsecutiveMatchCounter.record()`） | 不一致時に current_streak を 0 にするという事実を参照した。実装が正しい、または正式仕様であるとは判断しない |
| docstring（「0から数え直し」） | 出典とされる「Step6-1レビュー(d)」の原資料は確認できない（F5）。参考の記載として参照した |
| 既存のテスト | 現行の挙動が検証されているという事実を参照した。テストを正式仕様の根拠とはしない |
| Step5-0 §⑥ の「1件でも不一致ならNo-Go」 | 承認済みの項目ではなく（F2）、また No-Go についての記載である。本判断は、これを不一致時のリセットと同一視しない |
| Step6-3-52 §1・§10 | 現行実装の整理を判断材料として参照した。その整理をそのまま採用したものではない |
| Step6-3-69 §2.1 | 判定値の供給元の確定についての記載であり、本判断の対象（リセット）とは別の論点である（F8） |

### 5.3 UNRESOLVED（本判断で決めない事項）

- broken・broken_at の正式な意味、broken_at に何を記録するか（M10-U1・U3）
- max_streak の正式な役割（M10-U4）
- current_streak と max_streak のどちらを判定値とするか（M8-U4）
- 不一致と No-Go の関係（M8-U5 の後半）
- 一致判定の内容、比較段の範囲、段単位の未比較
- 一致時に current_streak を +1 する扱いの正式化

### 5.4 CONTRADICTION

なし。不一致時の current_streak の扱いは、既存資料に定義が書かれていない事項であり、資料間で相反する記載は確認されていない。Step5-0 の「1件でも不一致ならNo-Go」と現行実装のリセットは、別の概念についての記載である（M8〜M10正式定義突合 §2.3）。

---

## 6. 正式G2・G3 との関係

本判断は、次のいずれでもない。

- Step6-0 §⑤ の正式G2・G3 を変更する判断
- 正式G2・G3 の「100レース連続」の条件を再定義する判断
- G2改 を採用する判断
- G2・G3 の達成、GO・No-Go を判定する判断
- No-Go の条件を定める判断

---

## 7. 本判断で決めない事項

- broken・broken_at の意味、broken_at に何を記録するか
- max_streak の正式な役割
- current_streak と max_streak のどちらを判定値とするか
- 不一致と No-Go の関係、Go/No-Go の判定方法
- 一致判定そのもの、比較段の範囲、段単位の未比較
- skip の扱い（D2 で確定済みであり、変更しない）
- required=100
- 100レースの時間・実行の単位（M8-U1〜U3）
- Stage 3 の連続判定（C8）との関係（M8-U6）
- buy_decision の扱い、M10-U2
- G3-1、G3-3
- G2改、正式G2・G3
- Stage 3・Stage 5・Stage 6 の状態、GO・No-Go
- FROZEN とされている項目の解除
- 新しい完了条件、判定基準、Go/No-Go 条件、工程番号

---

## 8. 判断後の状態

| 項目 | 状態 |
|---|---|
| D1（「1件」の粒度） | eval_id（＝対象レース）（D1記録） |
| D2（レース単位の skip） | streak の判定対象外。増加もリセットもしない（D2記録） |
| 不一致時の current_streak | 0 に戻し、その後は 0 から数え直す（本書） |
| D4 のその他の部分（max_streak の役割、判定値、不一致と No-Go の関係） | 未判断 |
| D3（broken） | 未判断 |
| 正式G2・G3（Step6-0 §⑤） | 既存の正式定義・承認状態を維持（本判断で変更なし） |
| M8-U1、buy_decision の streak 上の扱い | FROZEN のまま |
| Stage 3・Stage 5・Stage 6・GO | 各既存記録のとおり（本判断で変更なし） |

---

## 9. 既存記録との関係

本書は、継続基準書、Phase0.5 設計固定書 v1.1.8、Step5-0、Step6-0、Step6-3-52、Step6-3-69、その他の共通 Step 資料、P1P2記録、D1記録、D2記録、`docs/上位判断記録_G2_G3_Phase1_20260927.md`、`docs/上位監査記録_G2_G3状態整理_20261006.md`、M8〜M10 の各監査記録、その他の上位判断記録・上位監査記録の各件を改訂・置換しない。
