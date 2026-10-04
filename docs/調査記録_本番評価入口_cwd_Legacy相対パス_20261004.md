# 調査記録：本番評価入口におけるカレントディレクトリと Legacy 相対パス依存

作成日: 2026-10-04 ／ ステータス: **調査記録（既存資料・コードの照合結果の記録のみ・判定なし・設計決定なし）**

---

## §0 本書の位置づけ

- 本書は、Rebuild の本番評価入口（`rebuild/pipelines/production_evaluation_entry.py`）から Legacy が呼ばれる経路における、ファイル参照とカレントディレクトリ（以下「cwd」）の依存関係を調査した結果を記録するものである。
- 本書は、本番評価入口・実行環境の前提についての調査であり、Stage 3 の差分の原因調査ではない。`docs/Stage3_Shadow観測_原因調査記録.md`、`docs/Stage3_Shadow観測_原因調査記録_追補_20261002.md`、`docs/上位判断記録_Stage3_20260927.md` の内容・判断・FROZEN の状態は変更しない。
- 調査の対象は、既存のコード（origin/main `63dc1ca` 時点）、Workflow、正式記録、既存の実行記録（チャット上の記録）だけである。コードの実行・再実行、コードの変更、設計の変更は行っていない。
- 本書は、本番評価入口の起動場所、cwd、`PYTHONPATH`、パス解決の方式を決めるものではない。

### 本書で判断しないこと

- 本番評価入口の起動場所・cwd・`PYTHONPATH`・パス解決の方式
- 現状の配置やコードが不具合であるかどうか、修正が必要かどうか
- 本番 Workflow の仕様

---

## §1 正式資料から確認できたこと

| 資料 | 確認結果 |
|---|---|
| 継続基準書（`docs/再設計_完成基準・残作業・再開手順書.md`） | cwd、実行ディレクトリ、相対パス・絶対パスに関する規定はない |
| Phase0.5 設計固定書 v1.1.8（`docs/Phase0_5_設計固定書.md`） | cwd を固定する規定はない。§⑩（ディレクトリ構成固定・最終形、L497〜）は、`config/` の下に `asahi_config.json`・`buyscore_config.json`・`brand.json`・`delivery.json`・`pipeline.json` を置く形を示している（L538、§⑮ L708〜712）。評価データは `evaluations/{date}.jsonl`（L199・L343）。これらのパスが、リポジトリのどの位置または cwd を基準とするかは記載されていない |
| Phase1開始条件_正式判定記録 | cwd に関する規定はない。§5-3 は「設定ファイル5種のうち未作成分の整備」を Phase 1 以降の実装課題としている |
| `docs/Stage3_Shadow観測_原因調査記録_追補_20261002.md`（`7c24ef5`） | §3.1 に、Shadow が `rebuild/` を cwd として実行され、Legacy の設定・データファイル・モデル・fan ファイルを読めなかった事実が記録されている |

- cwd を固定する正式な規定は、確認できなかった。
- Legacy のファイルの配置を cwd との関係で規定する正式な資料は、確認できなかった。

---

## §2 コード上の確定事実

### 2.1 cwd を決める処理

- `rebuild/` のコード（テストを除く）に、cwd を変更する処理（`os.chdir`）、cwd を取得してパスを組み立てる処理、`__file__` を基準にパスを決める処理は存在しない。
- 本番評価入口から Legacy までの経路（下記 2.2）のどの段階も、cwd を変更・固定・保証しない。cwd は、プロセスを起動した側が決める構造である。

### 2.2 本番評価入口から Legacy までの経路

| 段階 | 内容 | cwd の扱い |
|---|---|---|
| `rebuild/pipelines/production_evaluation_entry.py`（`main()`） | TARGET_RACES を受け取り、設定（締切・1日の上限）を読み、`run_production_day` を呼ぶ | 変更・固定しない |
| `rebuild/actions/production_entrypoint.py`（`build_production_bundle`、`run_production_day`） | Shadow の `build_bundle()` をそのまま再利用して部品を構築する（L361〜382） | 変更・固定しない |
| `run_one_race()`（同ファイル） | 評価・Buy・Output・通知リクエストの生成 | 変更・固定しない |
| Legacy の関数 | 出走表・オッズの取得、`_evaluate_bets`、`apply_buyscore`、設定の読み込み、場の統計など | 呼ばれた時点の cwd からの相対パスでファイルを読み書きする |

### 2.3 Rebuild 側の相対パス

| ファイル | 参照元 | 備考 |
|---|---|---|
| `config/pipeline.json`・`config/delivery.json` | `rebuild/actions/config_loader.py` L21（`CONFIG_DIR = "config"`）・L64・L72 | 見つからない場合は `ConfigError`（L36〜37）。本番評価入口は `ConfigError` で終了コード 1 を返す（`production_evaluation_entry.py` L263〜268） |
| `evaluations/{date}.jsonl` | `rebuild/pipelines/production_evaluation_entry.py` L61〜62 | 読み書き |
| `notification_counts/{date}.json` | `rebuild/actions/notification_counter.py` L43 | 日次上限の設定時に読み書き |
| `system_metrics.json`・`metrics/` | `rebuild/pipelines/production_evaluation_entry.py` L148〜149 | GitHub Actions 上（`GITHUB_RUN_ID` あり）で書き込み |

### 2.4 Legacy 側の相対パス（本番評価入口の経路で使われるもの）

| ファイル | 参照元 |
|---|---|
| `asahi_config.json` | `x_asahi_scoring.py` L45（`load_asahi_config`）← `rebuild/actions/shadow_entrypoint.py` `_load_freeze_configs`（`build_bundle` 経由） |
| `buyscore_config.json`・`buyscore_config.default.json` | `x_buyscore.py` L33〜34（`load_config`・`_ensure_config_file`）← `_load_freeze_configs`、`apply_buyscore` |
| `motor_history.csv` | `x_venue_stats.py` L40・L106 ← `rebuild/features/feature_builder.py` |
| `hit_record.csv` | `notify_arashi.py`（`_evaluate_bets` の中） |
| fan ファイル（`fan*.txt`） | `notify_arashi.py` L642〜644（`_get_fan_file`、`glob`）← `_extract_boats_from_program`（L982） |
| `model_all.pkl` | `notify_arashi.py`（`_load_ml_model`）← `_predict_win_prob` |
| `buyscore_log.jsonl`（書き込み） | `x_buyscore.py` L592（`save_buyscore_log`） |

参考：`daily_stats.json` 等（`x_results_common.py` ← `generate_public_html`）は、Renderer を使う構成（`build_production_bundle()`）では使われるが、本番評価入口（Renderer なし）では使われない。

### 2.5 import

- 本番評価入口は `actions.*`・`pipelines.*` など `rebuild/` の下のパッケージを import し、Legacy は `notify_arashi` などリポジトリ直下のモジュールを import する。
- リポジトリ直下に `models.py` があり、`rebuild/models/` と名前が重なる。

---

## §3 配置上の事実

| 対象 | git 上の位置 |
|---|---|
| Rebuild の設定（`pipeline.json`・`delivery.json`） | `rebuild/config/` だけ（リポジトリ直下に `config/` はない） |
| Legacy の設定（`asahi_config.json`・`buyscore_config.json`・`buyscore_config.default.json`） | リポジトリ直下 |
| Legacy のデータ・モデル（`motor_history.csv`・`model_all.pkl`・`fan2604.txt`） | リポジトリ直下 |
| `hit_record.csv`・`buyscore_log.jsonl` | git 管理外 |

- 現在の配置では、Rebuild の設定（`rebuild/config/`）と、Legacy の主要なファイル（リポジトリ直下）を、1つの cwd から相対パスで同時に解決することはできない。
- 本書は、この状態を不具合とは判定しない。

---

## §4 Shadow との違い

| 項目 | Shadow | 本番評価入口 |
|---|---|---|
| cwd | `.github/workflows/shadow_run.yml` で `rebuild/` に固定 | 固定する Workflow はない |
| `PYTHONPATH` | 同 Workflow でリポジトリ直下を指定 | 定めた資料はない |
| Legacy のファイル | `rebuild/` には存在せず、読めなかった（`7c24ef5` §3.1） | 起動場所による |
| Rebuild の `config/` | `rebuild/` から見つかる位置 | 起動場所による |

**既存の実行記録（2026-09-23、チャット上の記録）**

- 本番評価入口は、リポジトリ直下（`C:\Users\81809\my-app`）を cwd とし、`PYTHONPATH` にリポジトリ直下と `rebuild` を指定して実行された（Claude の案内による手順）。
- 最初の `python -m pipelines.production_evaluation_entry` は `ModuleNotFoundError: No module named 'models.evaluation'; 'models' is not a package` で失敗し、`sys.path` の先頭に `rebuild` を入れる形で実行し直された。
- この実行では、Legacy が `motor_history.csv` を取得して場の統計を算出し、評価データはリポジトリ直下の `evaluations/20260922.jsonl` に保存された（`6ced1b9`、2026-09-23 14:25）。
- この実行は、設定の読み込み（`config_loader`）を追加した commit `3628562`（2026-09-23 21:53）より前である。

---

## §5 未確定の事項

- 本番評価入口の正式な起動場所
- 正式な `PYTHONPATH` の前提
- Rebuild 側と Legacy 側の相対パスを、どのように両方成立させるか
- commit `3628562` 以降の、本番評価入口の実行実績
- import 名の重なり（リポジトリ直下の `models.py` と `rebuild/models/`）の正式な扱い
- Phase0.5 §⑩ の `config/`・`evaluations/` などのパスが、リポジトリのどの位置を基準とするか

---

## §6 判定

### 6.1 確定事実

- 本番評価入口の経路には、Rebuild 側・Legacy 側の両方に cwd 依存の相対パスが存在する。
- 本番評価入口の側に、cwd を固定する仕組みはない。
- 現在の配置では、Rebuild の設定（`rebuild/config/`）と Legacy のリポジトリ直下のファイルを、同一の cwd から同時に解決することはできない。

### 6.2 USER DECISION

- 本番評価入口をどこから起動するか
- cwd・`PYTHONPATH`・パス解決の方式を、どのように正式化するか

これらは、本番 Workflow の仕様（S1〜S7 の後続設計事項。S2〜S5・S7 は原文根拠不足として FROZEN）とも関係する。

### 6.3 記録しないこと

- 現状が不具合であるという評価
- 修正が必要であるという評価
- 起動場所・cwd・パス解決の方式の案

---

## §7 既存記録との関係

- 本書は、既存の正式記録（継続基準書、Phase0.5 設計固定書、Phase1開始条件_正式判定記録、上位判断記録の各件、Stage 3 の原因調査記録・追補）を改訂・置換しない。
- Stage 3、S6、G2・G3・G5、§591・§595〜597、IdempotencyStore、C5 などの既存の判断に触れない。
- 新しい完了条件・工程番号・優先順位は定義しない。
