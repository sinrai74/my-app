# Phase1開始条件_正式判定記録

作成日: 2026-09-26 ／ ステータス: **正式判断（Phase 0.5 §⑱ No.1〜No.14 の判定のみ・実装なし・実測なし）**

---

## §0 本書の位置づけ

- Phase 0.5 設計固定書 v1.1.8 §⑱「Phase 1 開始条件」の14項目について、**各条件文を基準に現時点の判定を記録する**。
- 判定基準は §⑱ の条件文のみとする。新しい完了条件・判定基準・例外ルールは作らない。
- 設計書本文（⑱のチェック欄を含む）は変更しない。本書は独立した判定記録である
  （既存前例: `Step6-3-69_O-8最終判断.md`、`docs/incidents/2026-07-13_hit-record-gap-unrecoverable.md`）。

### 本書で判断しないこと

- **Phase 1 開始そのものの可否**
- Phase 0.5 の設計内容の変更
- Feature Freeze（⑳）の変更
- S4 / S5.1 / S5.2 / S6 の実装変更
- G1〜G8（Go/No-Go）の判定
- Step体系とPhase体系の対応関係の新規定義
- No.14 未充足資産の保全方法

---

## §1 判定の前提

| 項目 | 内容 |
|---|---|
| 判定基準 | Phase 0.5 設計固定書 v1.1.8 §⑱ の14条件文 |
| 正式採用版 | `Phase0_5_設計固定書_v1_1_8.md`（版 1.1.8 ／ 状態: 確定（Frozen）） |
| 正式ファイル | `docs/Phase0_5_設計固定書.md`（commit `aa17242`） |
| 旧版の扱い | docs内に並置しない。追跡性はGit履歴で担保 |

### 判定区分

- **充足**: 条件文が要求する内容が、既存資料・実体から確認できる
- **未充足**: 条件文が要求する内容が満たされていないことが確認できる

### 判定の共通方針

条件文が求めるのは「**確定している**」「**文書化されている**」であり、対応する実装が完了していることではない。
したがって、実装が未完了であることのみを理由に条件を未充足とはしない。残る実装課題は §5 に分離して記録する。

---

## §2 判定一覧

| No. | 条件（要旨） | 判定 |
|---|---|---|
| 1 | 本設計書（1.1版）が docs/ にコミットされ、確定版として承認 | **充足** |
| 2 | データモデル（③）の全項目・型・必須が確定し、追加の未決事項がない | **充足** |
| 3 | EvaluationEngine / BuyEngine インターフェース（⑤5.2）が確定 | **充足** |
| 4 | レイヤー責務（⑤〜⑧）と依存ルール（⑩）が確定 | **充足** |
| 5 | 命名規則（⑪）と既存名の移行対応表が確定 | **充足** |
| 6 | ディレクトリ構成（⑩）が確定 | **充足** |
| 7 | ログ形式（⑬）とエラー方針（⑫）が確定 | **充足** |
| 8 | テスト戦略（⑭）の一致基準・Phase切替条件が数値で確定 | **充足** |
| 9 | 設定ファイル5種（⑮）の構成が確定 | **充足** |
| 10 | 移行方法・バックアップ・ロールバック（⑥⑯）が文書化 | **充足** |
| 11 | Feature Freeze（⑳）が宣言され、凍結タグが打たれている | **充足** |
| 12 | ゴールデンデータ100レース分の入力データが確保 | **充足** |
| 13 | hit_record.csv 欠落期間の復旧可否が確定 | **充足** |
| 14 | 現行システムの全資産（Phase 0 ⑨）がReleasesまたはgitで保全済み | **未充足** |

---

## §3 各判定の根拠

### No.1 充足

- 条件は2要素から成る。
  - (a)「リポジトリ docs/ にコミットされ」→ **commit `aa17242`「docs: Phase 0.5 設計固定書 v1.1.8 を正式採用版として追加」により `docs/Phase0_5_設計固定書.md` として登録済み**（868行）。origin/main へ push 済み。
  - (b)「確定版として承認されている」→ v1.1.8 のヘッダに「状態: **確定（Frozen）**」、⑳ L851 に「開始: 2026-07-13（**本書承認日**）」。加えて v1.1.8 を正式採用版とする決定が存在する。
- 配置ファイルの内容は v1.1.8 と同一（MD5 `c28bf1ad162699469b5bface6be07e4f`）。本文の編集・版番号の変更・過去版の併置は行っていない。

### No.2 充足

- ③3.1 に `close_time | str | 必 | 締切時刻 HH:MM` と定義され、型・必須・値域が確定している。
- FeatureSet(3.3)・SystemMetrics(3.16) は v1.1 で追加済み。改訂履歴 1.1.3〜1.1.6 により Optional契約・型の未決事項は順次確定済み。
- close_time の実データ（日時形式）との差異は、⑦ L92 が L0 data 層の責務と定める「データモデルへの正規化」が未実装であることによる。**モデル契約の未決事項ではない**（§5-1）。

### No.3 充足

- ⑤5.2 に EvaluationEngine / BuyEngine のインターフェースが定義されている。改訂履歴上、1.1 以降に 5.2 を変更した記録はない。

### No.4 充足

- ⑤〜⑧に各層の責務（L92「L0 data: 外部ソースからの取得・パース・リトライ・データモデルへの正規化」等）、⑩に依存ルールが定義されている。
- 改訂履歴 1.1.2・1.1.7 は実装との差異を「実態へ是正し記録」した更新であり、設計側の未決事項は確認されない。

### No.5 充足

- ⑪ の章題が「**命名規則（確定）**」であり、L607 に既存名の移行対応表が記載されている。

### No.6 充足

- ⑩にディレクトリ構成が定義され、1.1.2・1.1.7 で実装実態へ是正済み。

### No.7 充足

- ⑫に例外6分類（DataFetchError / ParseError / StorageError / DeliveryError / ConfigError / ModelError）とリトライ規定、⑬にログ形式・レベルが定義されている。

### No.8 充足

- ⑭ に「一致基準（**Phase切替の合格ライン・確定値**）」として数値が定義されている
  （danger_score 100%、upset_score 順位相関0.99以上、rank_index・featured_boats 100%、BuyScore・purchased判定 100%、pred_combo・patterns 100%、hit_record 44互換列100%（浮動小数1e-6許容））。

### No.9 充足

- ⑮ に設定ファイル5種（`asahi_config.json` / `buyscore_config.json` / `brand.json` / `delivery.json` / `pipeline.json`）の具体名・構成・`_version`・起動時スキーマ検証が定義されている。

### No.10 充足

- ⑥に保存対象・保存タイミング・バックアップ・復旧方法の表、⑯にバージョン管理方針が記載されている。条件は「文書化されている」であり、文書は存在する。

### No.11 充足

- `docs/freeze_baseline.md`（凍結宣言日 2026-07-13、根拠 v1.1 ⑳20.4）、gitタグ `freeze-v1-baseline`、commit `d258757`。

### No.12 充足

- `rebuild/tests/regression/golden4/manifest.json`: `count: 100`、`baseline_tag: freeze-v1-baseline`、inputs / expected が実在。

### No.13 充足

- `docs/incidents/2026-07-13_hit-record-gap-unrecoverable.md` に、20260705〜20260707 の3日分について調査結果と「**復旧不能確定**」が記録されている（commit `e316d78`）。

### No.14 未充足

§4 に記載する。

---

## §4 No.14 の判定根拠

**条件**: 現行システムの全資産（Phase 0設計書⑨）がReleasesまたはgitで保全済みである

**判定基準**: Phase 0 §⑨ の17資産を対象とする。Phase 0 §⑩ の完了条件「**⑨の資産すべてがReleasesまたはgitで保全されている**」を要件とする。保全先は「**Releases または git**」に限定され、それ以外の場所（プロジェクト資料領域＝common files 等）は本条件の保全先に含まれない。

### 4.1 実体確認の結果（2026-09-25）

リモートのタグは3つのみ。資産用の別Releaseは存在しない。

| タグ | commit | 内容 |
|---|---|---|
| `data-store` | c2089b6 | Legacy運用データ永続化（⑨-4 の対象） |
| `data-store-v2` | 7b23cc8 | Rebuild が使用 |
| `freeze-v1-baseline` | d258757 | Feature Freeze 凍結タグ |

**`data-store` Release の asset（⑨-4 の対象）**

- `buyscore_log.jsonl`（147.8KB）／`buyscore_log.jsonl.bak`（47.4KB）
- `daily_stats.json`（76.5KB）／`daily_stats.json.bak`（76.3KB）
- `motor_history.csv`（506,396KB）／`motor_history.csv.bak`（351KB）
- **hit_record.csv・local_course_stats.csv・学習モデル・設定JSONは存在しない。**

**`data-store-v2` Release の asset**

- `evaluations_20260922.jsonl`（6.23KB）のみ。Rebuild の評価データであり、⑨の資産は含まれない。
- **⑨-4 の対象（`data-store`）の代替保全先とは扱わない。**

### 4.2 ⑨-1 hit_record.csv（全実績・44列スキーマ）＋schema_version管理の仕組み

- 現在の git HEAD に存在しない（`.gitignore` L16 で除外。commit `469d2dc`（2026-07-04）で166行が削除され管理対象外化）。
- Releases（`data-store`・`data-store-v2`）に存在しない。PCローカルにも存在しない。
- git履歴から取り出せる最終状態は commit `8d2752a`（2026-07-03）時点の **166行＝165件・22列・対象期間 20260628〜20260703**。
- commit `2020772`（2026-07-01）「重複レコードを除去（1486件→26件）」について、除去前 `5225aad` の1,492行を検査した結果、`date+venue_num+race` のユニークキーは26件であり、除去後の26件と**キー集合が完全に一致**する（差集合は空）。したがって当該削除は重複の解消であり、**実績レースキーの欠落は生じていない**。
- 22列版を現行の44列スキーマへ変換する**正式な移行仕様は存在しない**。`rebuild/storage/schema/hit_record_migration.py` は「レガシー**44列** → 新形式（44+拡張列）」の1方向のみを対象とし、ヘッダが44列でも拡張形式でもない場合は StorageError（破損検知）とする。`hit_record_mapper.py` も44互換列の欠落を ParseError とする。
- 20260705〜20260707 の3日分の復旧不能（`docs/incidents/2026-07-13_hit-record-gap-unrecoverable.md`）は **No.13 の条件（欠落期間の復旧可否の確定）に関する事項であり、本条件（資産の保全）とは別問題**である。
- `rebuild/tests/regression/golden/hit_record_golden_100.csv`（44列・100件）は ⑭ の回帰試験用データであり、**⑨-1 の「全実績」の代替としない**。
- ⑨-1 の後半「schema_version 管理の仕組み」は git 上に保全済み（`hit_record_mapper.py`・`hit_record_repository.py`・`hit_record_migration.py`）。

**結論: ⑨-1 の「全実績」本体は Releases・git のいずれにも保全されていない。**

### 4.3 ⑨-3 local_course_stats.csv（205,278行の当地コースDB）＋k_race_history進捗ファイル

- **git履歴に一度も存在しない**（`git log -- local_course_stats.csv` が0件。`.gitignore` にも記載がなく、除外指定によるものではない）。
- Releases（`data-store`・`data-store-v2`）にも存在しない。PCローカルにも存在しない。
- common files に同名ファイル（205,278行）が存在するが、**common files に存在することと、本条件の保全先（Releases または git）で保全済みであることは別である**。

**結論: 未保全。**

### 4.4 ⑨-10 x_kfile_race_parser / lzh_extract

- `x_kfile_race_parser.py` は git 上に存在（保全済み）。同ファイルに `lzh` の文字列はなく、**lzh_extract の統合は確認されない**。
- `lzh_extract.py` は **git履歴に一度も存在せず**、Releases にもなく、PCローカルにも実体がない。
- `download_k_history.py`（git管理下）が `import lzh_extract` し、`extract_all()` / `BadLzhFile` を使用している。したがって**現在のgit管理コードだけでは当該スクリプトの依存が満たされない**。
- common files に同名ファイルが存在するが、4.3 と同じく本条件の保全先ではない。

**結論: 本項目は未保全（2ファイル中1ファイルが不在）。**

### 4.5 No.14 の判定

⑨-1・⑨-3・⑨-10 が Releases または git で保全されていないため、Phase 0 §⑩ の完了条件「⑨の資産すべて」を満たさない。

**判定: 未充足。**

---

## §5 Phase 1 以降の実装課題（条件の未充足ではない）

| # | 内容 | 関連 |
|---|---|---|
| 1 | L0 Adapter で `close_time` を ③3.1 の形式（HH:MM）へ正規化する。その際、S4.1 の締切判定（`CLOSE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"`）との整合性を併せて確認する | No.2 / S4.1 |
| 2 | 例外6分類のうち未実装のクラス（DataFetchError / DeliveryError / ModelError）の実装 | No.7 / ⑫ |
| 3 | 設定ファイル5種のうち未作成分の整備 | No.9 / ⑮ |
| 4 | 設計書 ⑩ に記載され未実装のディレクトリ・モジュール（`services/` 等） | No.4 / No.6 |
| 5 | S6 未実装項目: 除外理由record（保存先・モデル・形式・保持期間・粒度・分類は未決定） | S6 |
| 6 | hit_record.csv を GitHub Releases の退避対象に含めること（`docs/incidents/2026-07-13_…` 対応方針3が Phase 1以降の storage層設計で担保するとしている） | ⑥ |

---

## §6 未判断事項

本書では以下について結論を出さない。

1. common files の版を正式原本として認定するか
2. `local_course_stats.csv` を git へ追加するか
3. `lzh_extract.py` を git へ追加するか
4. `hit_record.csv` をどう扱うか
5. 欠損資産を保全して No.14 を充足させる方針を採るか
6. No.14 未充足のまま Phase 1 を開始できるか
7. Phase 1 開始そのもの

No.14 が未充足であることを理由に、新しい条件・例外ルールは作らない。

---

## §7 本書の結論

- **No.1〜No.13: 充足**
- **No.14: 未充足**（⑨-1 hit_record.csv・⑨-3 local_course_stats.csv・⑨-10 lzh_extract.py が Releases・git のいずれにも保全されていないため）

§⑱ は「1つでも未確定なら Phase 1 に着手しない」と定める。本書は No.14 が未充足であるという事実を記録するものであり、**Phase 1 開始の可否そのものは判断しない**。

---

## §8 変更していないもの

- Phase 0.5 設計書本文（⑱のチェック欄を含む）
- Feature Freeze（⑳）
- S4 / S5.1 / S5.2 / S6 の実装
- Legacy コード / Shadow 関連
- G1〜G8 の判定
- ⑨の17資産（保全作業は実施していない）

---

# 【正式更新】2026-09-26 ⑨-3・⑨-10 保全完了と F-1 判断の確定

ステータス: **正式判断（No.14 の対象資産に関する更新のみ）**

本節は §0〜§8 の判断（2026-09-26 作成時点）を削除・改変せず、その後に実施した保全作業と
その結果による判定の変化を追記するものである。本節の記載が現在状態を示す。

## 9.1 F-1 正式判断（hit_record 残存データの扱い）

git履歴 `8d2752a` から確認できる 165件・22列（対象期間 20260628〜20260703）の hit_record について、
以下を正式に確定した。

- **「現存する歴史的残存データ」と位置づける。**
- ⑨-1 が要求する「全実績・44列スキーマ」を満たす**正式な現行 hit_record とは認定しない**。
- **22列→44列の変換は行わない**（正式な移行仕様が存在しないため）。
- **再構成・推定・補完は行わない**（`docs/incidents/2026-07-13_…` 対応方針1と整合）。
- **原形を維持**して保全する（列追加・値変更を行わない）。
- 現行運用の `hit_record.csv` と**同一 asset 名では保存しない**。
- 既存 `data-store` Release へ、asset 名 **`hit_record_legacy22_20260703.csv`** として保全した。
- 本 asset の追加は Phase0.5 §⑥ L439「表にない新ファイルの作成はレビュー必須」に該当するため、
  **本 F-1 判断をもってその追加を承認する**。
- **この保全によって ⑨-1 の要件を満たしたとは扱わない。**

## 9.2 今回の保全実績

### ⑨-3 local_course_stats.csv — 保全完了

| 項目 | 内容 |
|---|---|
| 保全先 | 既存 `data-store` Release（新しい Release / tag は作成していない） |
| asset 名 | `local_course_stats.csv` |
| 件数 | 205,278件（ヘッダ除く） |
| 列 | 15列（racer_no, venue_code, venue_name, course, starts, first..sixth, first_rate, top2_rate, top3_rate, last_updated）。Phase0.5 §⑥ L263 の固定列と完全一致 |
| サイズ | 10,017,001 bytes（Release表示 9.55 MB） |
| 内容 | 原形のまま。`last_updated` が全行 0 である点も含め、値の修正・再計算・再構成は行っていない |
| `.bak` | `local_course_stats.csv.bak` が同時に生成された。これは `x_release_storage.upload_file()` の既存の世代交代処理によるものであり、初回アップロードのため本体と同一 sha256 である |
| git | `.gitignore` の方針（運用データは git に置かない）に従い、**git commit には含めていない** |

保存先の根拠: Phase0.5 §⑥ L431「local_course_stats.csv ／ 週次再構築 ／ 全置換 ／ **構築毎にReleases**」。

### ⑨-10 lzh_extract.py — 保全完了

| 項目 | 内容 |
|---|---|
| 保全先 | git（リポジトリ直下） |
| commit | **`00d8b04`**「chore: lzh_extract.py（Phase0 ⑨-10 ロジック資産）をGit管理下に追加」（1ファイル・448行） |
| 公開API | `extract_all(data: bytes) -> dict[str, bytes]`（L433）／ `BadLzhFile`（L31） |
| 依存 | 標準ライブラリのみ（外部Cライブラリ不要） |
| 整合性 | `download_k_history.py` が `import lzh_extract` し `extract_all()` / `BadLzhFile` を使用する呼び出しと API が一致 |
| Legacy | `download_k_history.py` を含む Legacy コードは**変更していない**（新規ファイルの追加のみ） |

保存先の根拠: ⑨-10 はロジック資産であり、`.gitignore` の方針（ソースコードは git に保存）に合致する。

### hit_record 残存データ — 保全完了（⑨-1 の充足とは別）

| 項目 | 内容 |
|---|---|
| 保全先 | 既存 `data-store` Release |
| asset 名 | `hit_record_legacy22_20260703.csv` |
| 件数 | 165件 |
| 列 | 22列（date, venue, venue_num, race, night, race_type, why_bet, confidence, pred_combo, pred_prob, pred_ev, pred_odds, upset_score, wind_speed, wind_dir, wave, result_combo, payout, hit, profit, n_bets, cost） |
| 対象期間 | 20260628〜20260703 |
| サイズ | 26,185 bytes（Release表示 25.6 KB） |
| 出所 | git履歴 `8d2752a`（2026-07-03）時点の hit_record.csv を原形のまま抽出 |
| `.bak` | `hit_record_legacy22_20260703.csv.bak` が同時生成。`upload_file()` の既存世代交代処理によるもので、初回のため本体と同一 sha256 |
| 現行 hit_record.csv | **追加していない**。`data-store` に `hit_record.csv` は存在せず、現行の世代管理には混入していない |
| 変換 | 22列→44列の変換は行っていない |

## 9.3 ⑨17資産の再確認（2026-09-26）

| ⑨ | 資産 | 保全先 | 状態 | 今回の変化 |
|---|---|---|---|---|
| 1 | hit_record.csv（全実績・44列スキーマ）＋schema_version管理の仕組み | ― | **未充足** | 残存データを別名で保全したが、要件は未達（9.4） |
| 2 | motor_history.csv | git ＋ Releases | 保全済み | ― |
| 3 | local_course_stats.csv ＋ k_race_history進捗 | Releases | **保全完了** | **未保全 → 保全完了** |
| 4 | Releases上の全アーカイブ | `data-store` | 保全済み（Release実在・asset 10件） | asset 4件増 |
| 5 | buyscore_log.jsonl / daily_stats.json | Releases（daily_stats.json は git にも） | 保全済み | ― |
| 6 | x_asahi_scoring.py | git | 保全済み | ― |
| 7 | x_buyscore.py | git | 保全済み | ― |
| 8 | x_ranking.py | git | 保全済み | ― |
| 9 | notify_arashi.py（fetch_programs・FANパーサー） | git | 保全済み | ― |
| 10 | x_kfile_race_parser / lzh_extract | git | **保全完了** | **一部未保全 → 保全完了** |
| 11 | x_release_storage.py | git | 保全済み | ― |
| 12 | x_verification.py | git | 保全済み | ― |
| 13 | asahi_config.json / buyscore_config.json | git | 保全済み | ― |
| 14 | scoring_spec.md / local_course_stats_design.md | git | 保全済み | ― |
| 15 | requirements-ml.txt | git | 保全済み | ― |
| 16 | ymlのcron設計 | git | 保全済み | ― |
| 17 | 「朝刊のみ」ポリシー | 設計書 | 保全済み | ― |

## 9.4 ⑨-1 を未充足のまま維持する理由

- ⑨-1 が要求するのは「hit_record.csv（**全実績**・**44列スキーマ**）」である。
- 今回保全した `hit_record_legacy22_20260703.csv` は **22列・165件・6日分（20260628〜0703）**であり、
  全実績でも 44列スキーマでもない。
- 22列を44列へ変換する正式な移行仕様は存在しない（`hit_record_migration.py` は「レガシー44列 → 新形式」の
  1方向のみを対象とし、44列でも拡張形式でもないヘッダは StorageError とする）。
- 20260704 以降の実績は git・Releases・ローカルのいずれにも存在しない。20260705〜0707 は
  `docs/incidents/2026-07-13_hit-record-gap-unrecoverable.md` により復旧不能確定。
- `rebuild/tests/regression/golden/hit_record_golden_100.csv`（44列・100件）は ⑭ の回帰試験用データであり、
  全実績の代替ではない。
- したがって、**「165件22列の歴史的残存データを別名 asset として保全したこと」と「⑨-1 が要求する
  全実績・44列スキーマを満たすこと」は別問題である。**
- なお ⑨-1 後半の「schema_version 管理の仕組み」は git 上に保全済み（`hit_record_mapper.py` /
  `hit_record_repository.py` / `hit_record_migration.py`）。

## 9.5 No.14 の正式判定（更新後）

⑨-3 と ⑨-10 は保全完了したが、**⑨-1 が未充足**であるため、Phase 0 §⑩ の完了条件
「⑨の資産すべてが Releases または git で保全されている」を満たさない。

**No.14: 未充足（維持）**

⑨-3・⑨-10 が解消されたことのみを理由に No.14 を充足へ変更しない。
本節は Phase 1 開始の判断ではなく、Phase 1 は開始していない。

## 9.6 更新後の判定一覧

| No. | 判定 |
|---|---|
| 1 | 充足（commit `aa17242`） |
| 2〜13 | 充足 |
| 14 | **未充足**（⑨-1 のみ未達） |

## 9.7 本節で実施していないこと

- Phase 1 の開始、および開始可否の判断
- ⑨-1 を充足とする解釈
- hit_record の 22列→44列 変換、欠損期間の推定・再構成
- Legacy コード、Feature Freeze 対象（評価式・スコア・スキーマ）、Shadow comparator / runner の変更
- 新しい Release / tag の作成
- 新しい完了条件・Step の作成

---

## 変更履歴

| 更新日 | 更新内容 | 根拠 |
|---|---|---|
| 2026-09-26 | 初版作成。No.1〜No.13 充足、No.14 未充足として記録 | commit `aa17242`、Phase0 ⑨⑩、Phase0.5 §⑱、実体確認結果 |
| 2026-09-26 | §9 追記。F-1 判断確定、⑨-3・⑨-10 保全完了、⑨-1 未充足維持、No.14 未充足維持 | `data-store` Release の保全結果（`local_course_stats.csv` 9.55MB / `hit_record_legacy22_20260703.csv` 25.6KB）、commit `00d8b04`、Phase0 ⑨⑩、Phase0.5 §⑥ L263・L431・L439 |
