# Stage3_Shadow観測_原因調査記録_追補

作成日: 2026-10-02 ／ ステータス: **原因調査記録の追補（既存証跡とコードの照合結果の記録のみ・判定なし）**

---

## §0 本書の位置づけ

- 本書は、`docs/Stage3_Shadow観測_原因調査記録.md`（`6aa6158`、以下「原因調査記録」）の後に行った追加調査で判明した事実を、追補として記録するものである。
- 本書は上位判断記録ではない。新しい判断を含まない。
- 本書は、原因調査記録および `docs/上位判断記録_Stage3_20260927.md`（`7df5826`、以下「上位判断記録 Stage 3」）を改訂・撤回・置換しない。
- 上位判断記録 Stage 3 の B1 は、作成時点（2026-09-27）において「race_type 47件・`20260704_21_06` の upset_score は原因未確定」と記載している。本書は、その後の調査で得られた事実を追加するものであり、同記録の B1・B2・B3 の判断内容を変更しない。

### 本書で判断しないこと

- 差分を Rebuild の修正対象として採用すること
- Stage 3 の完了・未完了
- G2・G3 の達否、および正式受入条件の扱い
- 継続基準書 §11・§12 の扱い
- 新しい完了条件

---

## §1 調査対象

| 項目 | 内容 |
|---|---|
| 対象 Run | Shadow Run 36224354755（2026-09-26T06:38:20Z〜07:09:08Z） |
| 対象レース | 20260704 の90レース |
| 対象とした差分 | race_type（47件）、`20260704_21_06` の upset_score、`20260704_04_04` の購入 |

---

## §2 調査方法・参照した証跡

### 2.1 方法

- 既存のコード（origin/main）、既存の artifact、ユーザー提供の Shadow ログの読み取りのみ。
- コードの変更、Shadow の再実行、G2・G3 の再測定、Release の変更、新しいデータの取得は行っていない。
- asahi・buyscore の設定の既定値を確認するため、作業環境の一時フォルダ（設定ファイルが存在しない場所）で Legacy の読み込み関数（`x_asahi_scoring.load_asahi_config`、`x_buyscore.load_config`）を呼び、返された既定値を凍結版の設定ファイルと比較した。リポジトリのファイルは変更していない。

### 2.2 参照した証跡

| 証跡 | 内容 |
|---|---|
| ユーザー提供の Run 36224354755 Shadow ログ（`0_shadow.txt`、3,340行） | ジョブ全体のログ。L3321 の artifact の URL（`actions/runs/36224354755/artifacts/10900737629`）により対象 Run であることを確認 |
| `sent_20260704.txt` | Legacy の当日の記録（90件） |
| `rebuild/tests/regression/golden4/`（inputs・expected） | `20260704_21_06` の Golden の入力と期待値 |
| `.github/workflows/shadow_run.yml` | 実行時のカレントディレクトリ（`rebuild/`）、環境変数 |
| コード | 下記の各節に関数・行番号を記載 |
| `docs/上位判断記録_L439_hit_record保全_20261002.md`（`588b398`） | 44列の hit_record の照合結果 |

---

## §3 調査結果

### 3.1 Run 36224354755 の実行条件　【確定】

| 対象 | 確認した事実 | 確認の根拠 |
|---|---|---|
| `asahi_config.json` | 見つからず、既定値に切り替わった（「`[朝刊AI] asahi_config.json が見つかりません → デフォルト設定を使用`」） | ユーザー提供の Shadow ログ L275 |
| `motor_history.csv` | 存在せず、場の統計の算出がスキップされた（「`[venue] motor_history.csv が存在しないため統計算出をスキップ`」） | 同 L288 |
| `model_all.pkl` | 読み込みに失敗した（「`[legacy] [ML] モデル読み込み失敗: [Errno 2] No such file or directory: 'model_all.pkl'`」、134回） | 同 L292・L678 ほか |
| `GITHUB_TOKEN` | 設定されておらず、Releases の読み書きはすべてスキップされた（「`GITHUB_TOKEN存在=False GITHUB_REPOSITORY存在=True → 結果=False`」、190回） | 同 L276 ほか、実行ステップの env（L258〜270） |
| fan ファイル（`fan*.txt`） | **ログでは確認していない**（読み込みの有無を出力しない作りのため）。fan ファイルはカレントディレクトリで `glob("fan*.txt")` により探される。Shadow は `rebuild/` をカレントディレクトリとして実行し、git 上の fan ファイルはリポジトリ直下の `fan2604.txt` だけである。**このコード・リポジトリの構成・Workflow の定義から、Shadow の実行時に fan ファイルは読まれていないと導ける** | `notify_arashi.py` L642〜644、`shadow_run.yml`、git |
| asahi の既定値 | 計算に使用する値は、凍結版 `asahi_config.json`（SHA-256 先頭 `6a7862b8`）と128項目すべて同一。違いは説明用の注記2項目だけ | §2.1 の比較 |
| buyscore の既定値 | `rebuild/` に `buyscore_config.json` は存在せず、コード上は既定値が使われる（ログには出力されない作り）。計算に使用する値は、凍結版（SHA-256 先頭 `da6a4eda`）と55項目すべて同一 | §2.1 の比較、git |

**区別**：上の環境差が、差分の項目に影響する**経路を持つこと**と、差分の**原因であること**は別である。以下の各節では、両者を分けて記載する。

### 3.2 race_type（47件）　【差分の構造的理由を確定】

| 項目 | 内容 | 根拠 |
|---|---|---|
| Rebuild 側 | `RaceEvaluation.race_type` は常に空文字 `""`。コメントに「race_type分類（notify_arashi L1742 classify系）はGolden対象外のため本Stepでは未分類""とする」 | `rebuild/core/engine.py` L300〜302 |
| Legacy 側 | 比較に使われる値は、Shadow の中で再計算した値ではなく、当日の `sent_20260704.txt` の記録値 | `rebuild/shadow/legacy_source.py` L156 |
| 当日の記録 | 90件すべてに `race_type` の値がある | `sent_20260704.txt` |
| 環境差との関係 | `asahi_config.json`・`motor_history.csv`・`model_all.pkl` のいずれも参照しない | `rebuild/core/engine.py` |

- 以上により、比較されたレースでは race_type が必ず差分になる構造であり、比較された47件すべてで差分になったことと一致する。
- この差分は、Run 36224354755 の環境差では説明されない。
- 本書は、これを Rebuild の修正対象として採用するものではない。

### 3.3 `20260704_21_06` の upset_score　【一部確定・一部未確定】

**確定した事実**

| 事実 | 根拠 |
|---|---|
| Rebuild 側の `calculate_upset_score()` で upset_score を 0 にするのは、次の2つの分岐だけである。①1号艇の確率が0.65を超え、かつほかの艇より高い（L225）、②1号艇が A1 級（L230）。途中の確率の計算はロジット加算とシグモイド（`_add_effect`、L65）であり、それ以外で 0 にはならない | `rebuild/core/upset.py` |
| 「1号艇の確率」は `calc_boat_score()`（L69）から算出され（L151）、使うのは各艇の `avg_st`・`course_nyuko`・`course_st`・`motor`・`win_rate` と枠の重み、設定の `boat_relative_score` である。場の統計は使わない | `rebuild/core/upset.py` |
| 場の統計（`motor_history.csv` 由来）は `calc_danger_score()`（L161）にだけ渡され、upset_score には danger の値による倍率（L219〜222）として効く。倍率であるため、場の統計だけで upset_score が 0 になることはない | `rebuild/core/upset.py` |
| 各艇の `course_nyuko`・`course_st` は、Legacy の `_extract_boats_from_program()` が fan ファイルから設定する。`racer_class`・`win_rate`・`avg_st`・`motor` は出走表の API の値である | `notify_arashi.py` L982 付近 |
| 当日の Legacy の値（6.274）は、`calculate_upset_score_v2()` の結果（L3451）に、ML の確率による加点（`min(対抗艇のML確率 × 8.0, 4.0)`、L3465）を加えたものである | `notify_arashi.py` |
| Golden の値（7.2155）は、`x_asahi_scoring.build_race_evaluation_v4()` の結果である。Rebuild の `Ver4Engine` は、この系統を移植したものである | `rebuild/tools/golden_wrapper.py`、`rebuild/core/engine.py` |
| したがって、Stage 3 で比較した当日の Legacy の値と Rebuild の値は、別の計算関数の値である | 上記 |

**未確定の事項**

- Shadow でどちらの分岐（①・②）により 0 になったか（各艇の入力の値・確率がログにも artifact にも残っていない。Golden の入力では1号艇の級別は空欄だが、Shadow は出走表を取得し直しているため、同じとは限らない）
- fan ファイルが読まれず `course_st` がなかったことが、1号艇の確率を0.65より上にしたかどうか（経路はあるが、値が残っていない）
- 当日の Legacy の値（6.274）の計算条件の詳細（ML の読み込みの有無、入力の値）
- 当日の値と Golden の値の差の内訳

### 3.4 `20260704_04_04` の購入　【一部確定・一部未確定】

**確定した事実**

| 事実 | 根拠 |
|---|---|
| Shadow での経路：`PredictionProvider` → Legacy の `_evaluate_bets()` → その中で `apply_buyscore()` → `DefaultBuyDecisionBuilder` で BuyDecision にまとめる | `rebuild/shadow/prediction_provider.py`、`notify_arashi.py`、ユーザー提供の Shadow ログ L677〜699 |
| `model_all.pkl` がないため、`_predict_win_prob()` は空の `{}` を返す（`ml_probs={}`） | `notify_arashi.py` L288・L294〜295、ユーザー提供の Shadow ログ L678 |
| シナリオ計算は `has_exhibition and boats and ml_probs` のときだけ行われる（L2221）。Shadow では `has_exhibition` が未結線で既定値 `True` であり、`ml_probs` が空であるため、シナリオ計算は行われない | `notify_arashi.py` L2221、`rebuild/shadow/prediction_provider.py` L56、`rebuild/actions/shadow_entrypoint.py` L206 |
| `mc_probs` または `scenario_probs` が空の場合、確率の統合の重みは PL 0.55・MC 0・SC 0・市場 0.45 に固定される（L2331〜2332） | `notify_arashi.py` |
| このレースのログの「`レジーム: rough → PL=0.55 MC=0.00 SC=0.00 市場=0.45`」は、この固定の分岐と一致する | ユーザー提供の Shadow ログ L680 |
| 種を固定しない乱数（L2034）は、モンテカルロ計算（L2086）の中で使われる。このレースでは MC の重みが 0 であるため、確率の統合の段階では乱数の影響を受けない。モンテカルロの結果がほかの用途に使われるかは確認していない | `notify_arashi.py` |
| Shadow の Rebuild 側の結果は「`[購入判定] 購入: 平和島 4R 3点 (BuyScore最高=70)`」 | ユーザー提供の Shadow ログ L692 |
| 当日の Legacy の購入は2点・金額はいずれも0であった。44列の `hit_record.csv` では、20260704 の行に `retroactive_fix:amount_zero_ghost_purchase` のタグが付いている | 原因調査記録 §3.2、`588b398` §2 |

**未確定の事項**

- ML モデルがないことが、当日の Legacy との差分の原因であったか（当日の Legacy で ML が読めていたかの記録がない）
- 当日の Legacy の ML の実行条件
- 天候・展示・オッズ（Shadow では実行時に取得）などの入力の違いのうち、どれが効いたか
- 複数の要因によるものか
- 当日の購入（2点・金額0、後に `retroactive_fix` のタグが付いた状態）と、Shadow の Rebuild 側の購入（3点）を、点数で単純に比較してよいかという前提

---

## §4 原因確定／原因未確定の区別

| 項目 | 状態 |
|---|---|
| race_type（47件） | 差分の構造的理由を確定（Rebuild 側が常に空文字）。環境差では説明されない |
| `20260704_21_06` の upset_score | 0 になり得る分岐と、当日の値・Golden の値の計算関数の違いは確定。Shadow でどの分岐により 0 になったかは未確定 |
| `20260704_04_04` の購入 | Shadow で ML がないことにより確率の統合が固定の分岐になったことは確定。差分の原因は未確定 |

---

## §5 本書で実施していないこと

- コードの変更
- Shadow の再実行、G2・G3 の再測定
- Release の変更、新しいデータの取得
- 既存の正式記録（原因調査記録、上位判断記録 Stage 3、Stage 3 実測記録、継続基準書ほか）の改訂
- 差分を Rebuild の修正対象として採用すること

---

## §6 調査終了時点の状態

### 6.1 追加調査の扱い

既存の証跡だけで確定できる範囲は調査済みである。残っている因果（`20260704_21_06` が 0 になった分岐、`20260704_04_04` の購入の差の原因）を確定するには、Shadow の再実行、新しいデータの取得、または当日の実行条件を示す追加の証跡が必要である。現時点では、追加調査を **FROZEN** とする。

再調査の対象となるのは、次のいずれかが生じた場合である。

1. Shadow の実行時の入力・確率などの記録が新たに見つかった場合
2. 当日の Legacy の実行条件を示す既存の資料が見つかった場合
3. ユーザーが再実行・新しいデータの取得を明示的に判断した場合

### 6.2 正式判断への影響

本書により、次の事項は変更されない。

- Stage 3 は「完了判定不能」（上位判断記録 Stage 3 の B2）
- 差分に対して Rebuild の実装の修正へ進まない。継続基準書 §12 の二択を適用しない（同 B1）
- 継続基準書 §11・§12 を変更しない（同 B3）
- G2・G3 の正式判断（`b66ffe0`）
- 新しい完了条件を作らない
- 新しい修正対象を採用しない
