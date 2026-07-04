# ボートレースAI（notify_arashi）── 運用ドキュメント

長期運用を前提としたAI予想・購入判定・実績蓄積システム。
本READMEは「誰が運用しても同じ手順で管理できる」ことを目的とした
運用基盤のドキュメントです。

---

## 1. Git管理方針

### Git管理対象（コミットする）
- ソースコード（`*.py`）
- 設定ファイル（`buyscore_config.json`, `asahi_config.json`, `bitget_config.py` など）
- 設定テンプレート・ワークフロー定義（`*.yml`）
- ドキュメント（`README.md`, `scoring_spec.md`）
- サンプルデータ（動作確認用の小さな固定データのみ）

### Git管理対象外（コミットしない・`.gitignore`で除外）
- 実績CSV: `hit_record.csv`
- 学習用CSV: `motor_history.csv`
- 履歴CSV: `buy_history.csv`, `prediction_history.csv`, `verification_history.csv`
- マイグレーション自動バックアップ: `hit_record_backup_*.csv`
- スキーマバージョン記録: `.hit_record_schema_version`
- レース単位の一時ファイル: `pred_*.json`, `sent_*.txt`
- ログ: `logs/`, `*.log`, `notify_running.lock`
- キャッシュ・自動生成データ: `korogashi_cache.json`, `buyscore_log.jsonl`,
  `daily_stats.json`, `buyscore_analysis.json`
- Python生成物: `__pycache__/`, `*.pyc`
- 仮想環境: `venv/`, `.venv/`
- 一時ファイル: `*.tmp`, `*.bak`
- 秘匿情報: `.env`
- OS固有ファイル: `.DS_Store`, `Thumbs.db`

**理由**: 運用データは実行環境ごとに内容が異なり、日々増え続けるため、
Gitリポジトリに含めるとサイズ肥大・差分ノイズ・機密性の問題が生じます。
これらは各環境で `migration.py` によって管理します。

### 既にGit管理下にある運用データを外す場合
過去に誤って `hit_record.csv` 等をコミットしていた場合、`.gitignore` に
追加しただけでは追跡が止まりません。以下を一度だけ実行してください。

```bash
git rm --cached hit_record.csv motor_history.csv
git commit -m "chore: 運用データをGit管理対象外にする"
```

---

## 2. CSVスキーマ管理

`hit_record.csv` は列構成が「スキーマバージョン」として管理されています。
定義の正は `csv_schema.py` の1箇所のみです。

| バージョン | 列数 | 内容 |
|---|---|---|
| v1 | 22列 | 予想・結果・回収の基本項目のみ |
| **v2（現在）** | 34列 | v1 + 購入判定分離(`purchased`/`buyscore`/`match_index`/`skip_reason`) + Phase2学習用特徴量(`feat_*`) |

現在の期待バージョンは `csv_schema.CURRENT_SCHEMA_VERSION` で確認できます。

```bash
python -c "from csv_schema import CURRENT_SCHEMA_VERSION; print(CURRENT_SCHEMA_VERSION)"
```

### スキーマバージョンの記録場所
`hit_record.csv` 自体には管理用の列は追加しません（全行に冗長な値が入るため）。
代わりに同じディレクトリの `.hit_record_schema_version`（隠しファイル）に
現在のファイルの実バージョンを1行で記録します。

---

## 3. マイグレーション方法

### 起動時の自動チェック
`notify_arashi.py` の起動時（`run()` 冒頭）に自動でスキーマチェックが走ります。

1. `hit_record.csv` の存在確認
2. ヘッダー確認（既知のスキーマと一致するか）
3. `schema_version` 確認（期待バージョンと一致するか）
4. 一致しなければ自動マイグレーション → 変換後の整合性確認

**判定不能な状態（未知のヘッダーで、どのバージョンか機械的に決められない場合）は
サイレントに動作を続けず、実行を停止してエラーメッセージを表示します。**
この場合は下記の手動コマンドで状態を確認してください。

### 手動コマンド

```bash
# 現在の状態を確認するだけ（変換しない）
python migration.py --check

# 変換内容を表示するだけ（書き込まない）
python migration.py --dry-run

# 実際にマイグレーションを実行する
python migration.py
```

実行すると、以下の完了レポートが表示されます。

```
============================================================
 hit_record.csv マイグレーション結果
============================================================
 実行           : 完了
 総レコード数   : 165 件
 旧スキーマ     : v1（22列）
 新スキーマ     : v2（34列）
 追加した列     : 12 列
   - purchased              初期値="1"      補完 165 件
   - buyscore               初期値=（空欄） 補完 165 件
   ...
 バックアップ   : hit_record_backup_20260704_213040.csv
 整合性確認     : OK
 結果           : マイグレーション成功。
============================================================
```

### 将来スキーマを追加する手順（例: v2 → v3）
CSVを手作業で編集する運用は禁止です。必ず以下の3手順で対応してください。

1. `csv_schema.py` に v3 の完全な列リストと、新規列の初期値を追加する。
2. `migration.py` に `migrate_v2_to_v3(rows)` を実装する。
3. `migration.py` の `MIGRATIONS` に `(2, 3): migrate_v2_to_v3` を1行追加する。

これだけで、v1のファイルも v1→v2→v3 と自動で最新まで移行されます
（1段ずつのチェーン方式のため、途中バージョンを飛ばした変換は書きません）。

---

## 4. バックアップ

マイグレーション実行前に、必ず日時付きバックアップを自動作成します。

```
hit_record_backup_YYYYMMDD_HHMMSS.csv
```

- 書き込みまたは変換後の整合性チェックに失敗した場合、このバックアップから
  自動的に元のファイルへ復元されます（データは失われません）。
- バックアップファイルは `.gitignore` により管理対象外です。長期保管が
  必要な場合は、別途ストレージ（S3等）へ手動で退避してください。

---

## 5. データ整合性チェック

`data_integrity.py` が以下を検査します（`migration.py` 実行時、および
起動時チェック時に自動で走ります）。

- 列不足（期待スキーマにあるが実ファイルにない列）
- 列順不一致
- 重複列
- 未知の余分な列
- 型異常（数値であるべき列に数値でない値が入っていないか）

異常があれば `logging` 経由でログに出力されます（サイレントに握りつぶしません）。

---

## 6. 運用チェックレポート

現在の運用基盤の状態は `ops_check.py` でまとめて確認できます。

```bash
python ops_check.py
```

```
==================== 運用チェックレポート ====================
 Git管理対象      OK
 .gitignore       OK
 CSV Version      v2
 Migration        OK
 Backup           OK
 Schema Check     OK
 Data Integrity   OK
================================================================
```
