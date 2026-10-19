# incident（案）：hit_record.csv が Release に作られない／x_ranking の3ステップが保存データを読めない

**本書は未確定の文案（承認待ち）。docs/incidents/ には置いていない。採否はユーザーが判断する。**

記録日（案）: 2026-10-19 ／ ステータス: **修正前の記録**（修正は作業ブランチ `wip/asahi-stage1-20261018` の commit `78a8006`、main 未反映）

区分: **確認事実**＝コード・git 履歴で直接確認した ／ **推論** ／ **未確認**

基準: origin/main `fcf5c44`。

---

## 1. 事象

| # | 内容 | 区分 |
|---|---|---|
| E1 | Release `data-store` に `hit_record.csv` が無い（2026-10-18 に公開ページで確認した asset 12件に含まれない。`hit_record_legacy*` は別名の保管用） | 確認事実 |
| E2 | x_ranking.yml の Send - Verification・Send - Results Page・Send Note Report の3ステップの env に `GITHUB_TOKEN` が無い（`GMAIL_ADDRESS`・`GMAIL_APP_PASS` のみ）。同 workflow で `GITHUB_TOKEN` を渡しているのは Commit motor history のステップだけ | 確認事実（`fcf5c44` の x_ranking.yml） |

---

## 2. 原因

### 2.1 hit_record.csv が作られない（E1）

- `notify_arashi._check_yesterday_results`（L4972〜）は `sent_{前日}.txt` を読み（L4977〜4978）、ファイルが無いと処理を抜ける（L4988）。Release への保存（L5215）はこの後にある。
- GitHub Actions の実行環境は毎回新しく作られ、notify が書いた `sent_{当日}.txt`（L3857・L3955・L3990）を次の実行へ引き継ぐ処理が無い。このため、前日の sent は常に無く、保存（L5215）まで進まない。
- hit_record.csv の Release 保存（L5215）は commit `6c763ac`（2026-07-05）で追加された。2026-07-06 の commit `7c8390f` は「sent_{前日}.txt 早期return原因特定用のデバッグログを追加」している。
- 凍結タグ `freeze-v1-baseline` にも同じ構造がある（既存の調査 V-7。本書では再確認していない）。

### 2.2 x_ranking の3ステップが保存データを読めない（E2）

- `x_release_storage.download_file` は `is_available()`（`GITHUB_TOKEN` と `GITHUB_REPOSITORY` が揃っているか）が False だと、何も読まずに False を返す（main `fcf5c44` の L238〜240）。
- 3ステップで実行される `x_verification.py`・`x_results_page.py`・`x_note_report.py`・`x_buyscore_tuner.py --dry-run` は、hit_record.csv・buyscore_log.jsonl を `download_file` 経由で読む（x_verification L95・L393、x_results_common L852、x_buyscore_tuner L84・L112）。
- commit `6c763ac`（2026-07-05）の x_ranking.yml では、この3ステップ（と Update motor history・Generate rankings・Send Weekly Report）に `GITHUB_TOKEN` があった。commit `83069fb`（2026-07-09、「全時間帯cronを毎朝7:30(JST)の単一実行に統合」）で、これらのステップから `GITHUB_TOKEN` が無くなった。

---

## 3. 影響

| # | 内容 | 区分 |
|---|---|---|
| I1 | 2026-07-05 以降、Actions 上で hit_record.csv が記録されていない。検証・実績ページ・tuner の入力が無い | 確認事実（E1）・推論（期間） |
| I2 | E2 は現在、E1 に隠れている（Release に hit_record.csv が無いため、トークンがあっても読むものが無い）。E1 だけを直すと、Release には hit_record.csv ができるが、3ステップは読めないままになる | 推論（コードの流れ） |
| I3 | 3ステップのうち buyscore_log.jsonl を読む処理（x_results_common L852）は、今も Release にある購入ログを読めていない | 推論（コードの流れ） |
| I4 | 3ステップのメールは送られており、変わるのは本文の数字だけ | 推論（コード上、メール送信はデータの有無で止まらない） |

---

## 4. 未確認

- x_ranking の実行ログで `[DEBUG release_storage] is_available: GITHUB_TOKEN存在=False` が出ているか
- 実績ページ・検証メールの実際の本文
- 2026-07-05〜07-09 の間に、Actions 上で hit_record.csv が一度でも Release に保存されたか（現在の Release には無い）
- 2026-07-13 の incident の欠落（07-05〜07-07）と本件の関係（同じ原因によるものかは確認していない）

---

## 5. 対応（作業ブランチ `78a8006`、main 未反映）

- E1：`x_state_sync.py` を追加し、notify_arashi.yml の前後で前日の sent を Release から取り戻し、当日の sent を保存する。`notify_arashi.py` は変更しない。
- E2：3ステップの env に `GITHUB_TOKEN` を戻す（読み取りのみ。3ステップのスクリプトは Release に書かない）。
- `83069fb` で同じく無くなった Update motor history・Generate rankings・Send Weekly Report には戻さない。Generate rankings の `x_venue_stats`（L126）は Release から読むため、戻すと新聞の入力（motor_history）が変わりうる（⑭のスコア比較に影響しうる）。Weekly Report は `download_file` を使っていない。

---

## 6. 既存記録との関係

- `docs/incidents/2026-07-13_hit-record-gap-unrecoverable.md`：「副次的な発見（根本原因）」は「退避対象リストに hit_record.csv が含まれていなかった」と推定しているが、hit_record.csv の Release 保存はその前の commit `6c763ac`（2026-07-05）で追加済みである。本書は同記録を改訂・置換しない。同記録の対応方針3（新システムの storage 層の要件）にも触れない。
- `docs/incidents/2026-10-14_buyscore-log-duplicate-and-nonatomic-upload.md`：保存の非原子性（N-03）は同記録の範囲で扱う。本書は扱わない。
