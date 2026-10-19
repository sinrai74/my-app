# incident：hit_record.csv が Release に作られない／x_ranking の3ステップが保存データを読めない

記録日: 2026-10-19 ／ ステータス: **修正前の記録**
基準: origin/main `4461bd0`（2026-10-19 時点。`fcf5c44` からの差分は自動更新のデータファイル4つのみで、本書が引用するコードは同じ）
修正の候補: ブランチ `fix/storage-minimal-20261019` の commit `1f72842`（**main には未統合**）

区分: **確認事実**＝コード・git 履歴・GitHub Actions の実行記録で直接確認した ／ **推論**＝確認事実からの推定 ／ **未確認**

---

## 1. 事象

| # | 内容 | 区分 |
|---|---|---|
| E1 | Release `data-store` に `hit_record.csv` が無い（2026-10-18 に確認した asset 12件に含まれない。`hit_record_legacy*` は別名の保管用） | 確認事実 |
| E2 | `.github/workflows/x_ranking.yml` の Send - Verification・Send - Results Page・Send Note Report の3ステップの env に `GITHUB_TOKEN` が無い（`GMAIL_ADDRESS`・`GMAIL_APP_PASS` のみ）。同 workflow で `GITHUB_TOKEN` を渡しているのは Commit motor history のステップだけ | 確認事実 |

E1 と E2 は原因が別の、独立した2つの障害である（§2）。

---

## 2. 原因

### 2.1 E1：hit_record.csv が作られない

- `notify_arashi._check_yesterday_results`（L4972〜）は `sent_{前日}.txt` を読み（L4977〜4978）、ファイルが無いと処理を抜ける（L4988）。Release への保存（L5215）はこの後にある。
- GitHub Actions の実行環境は毎回新しく作られ、notify が書いた `sent_{当日}.txt`（L3857・L3955・L3990）を次の実行へ引き継ぐ処理は、notify_arashi.yml・notify_arashi.py のどちらにも確認できない（確認事実：コード）。引き継ぎが無い場合、前日の sent が無い実行では、`_check_yesterday_results` は保存（L5215）の前に処理を抜ける（確認事実：コード L4988）。
- 実行記録では、notify の実行ログ2件（2026-10-15 01:10 UTC 開始、2026-10-16 09:33 UTC 開始）で、`[DEBUG sent_file] path=sent_{前日}.txt exists=False size=-1` と `[DEBUG sent_file] return_route=A (ファイル不存在)` が出ている（確認事実：ユーザーが共有した実行ログ）。それ以外の実行で同じだったかは確認していない（未確認）。
- notify_arashi.py の hit_record.csv の Release 保存（L5215）は、commit `7c8390f`（2026-07-06 23:01 JST。メッセージは「debug: sent_{前日}.txt早期return原因特定用のデバッグログを追加(ロジック変更なし)」）で、`import x_release_storage` とともに追加された。保存の部品（`x_release_storage.py`）自体は commit `6c763ac`（2026-07-05）で追加された。
- 凍結タグ `freeze-v1-baseline` にも同じ構造がある（2026-10-18 の調査。本書では再確認していない）。

### 2.2 E2：x_ranking の3ステップが保存データを読めない

- `x_release_storage.download_file` は `is_available()`（`GITHUB_TOKEN` と `GITHUB_REPOSITORY` が揃っているか）が False だと、何も読まずに False を返す（L238〜240）。
- 3ステップで実行される `x_verification.py`・`x_results_page.py`・`x_note_report.py`・`x_buyscore_tuner.py --dry-run` は、hit_record.csv・buyscore_log.jsonl を `download_file` 経由で読む（x_verification L95・L393、x_results_common L852、x_buyscore_tuner L84・L112）。
- commit `6c763ac`（2026-07-05）の x_ranking.yml では、この3ステップ（と Update motor history・Generate rankings・Send Weekly Report）に `GITHUB_TOKEN` があった。commit `83069fb`（2026-07-09、「全時間帯cronを毎朝7:30(JST)の単一実行に統合」）で、これらのステップから `GITHUB_TOKEN` が無くなった。

---

## 3. 影響

| # | 内容 | 区分 |
|---|---|---|
| I1 | E1 により、検証・実績ページ・tuner の入力となる hit_record.csv が Release に無い。2026-10-18 時点の Release に hit_record.csv は無く、§2.1 の実行ログ2件でも `[release_storage] hit_record.csv のアセットが存在しません（初回起動）` が出ている。記録されなかった期間（hit_record.csv の Release 保存が加わった 2026-07-06 以降のいつからか）は、全実行のログを確認していないため確定しない | 確認事実（現在の Release・ログ2件）・未確認（期間） |
| I2 | E2 は現在、E1 に隠れている（Release に hit_record.csv が無いため、トークンがあっても読むものが無い）。E1 だけを直すと、Release に hit_record.csv はできるが、3ステップは読めないままになる | 推論（コードの流れ） |
| I3 | E2 により、3ステップのうち buyscore_log.jsonl を読む処理（x_results_common L852）は、今も Release にある購入ログを読めていない | 推論（コードの流れ） |
| I4 | 3ステップのメールは送られている。E1・E2 の修正で変わるのは本文の数字であり、送信の有無は変わらない | 推論（コード上、メール送信はデータの有無で止まらない） |

---

## 4. 実行時刻の記録（検証メールの対象日との関係）

GitHub Actions の実行記録（ステップ単位、JST）：

| 日付 | x_ranking：Send - Verification の開始〜終了 | notify：その日の最初の実行の Run upset detector の開始〜終了 |
|---|---|---|
| 2026-10-18 | 09:10:49〜09:10:51 | 09:34:17〜09:46:07 |
| 2026-10-19 | 09:22:21〜09:22:23 | 09:43:25〜09:51:54 |

- 両日とも、検証ステップは、その日の notify の最初の実行より前に終わっている（確認事実）。両日とも、JST 07:00 以降で検証ステップより前に notify の実行は無い（確認事実）。
- `x_verification` は JST の前日の行を読む（L56〜57・L87〜109）。前日の行を hit_record に書くのは、当日の notify の `_check_yesterday_results`（前夜の notify は前々日の行を書く）である（確認事実：コード）。
- 以上から、**修正の後も、実行時刻がこの2日と同じ並びであれば、検証ステップが読む時点には前日の行がまだ無く、検証メールの件数は変わらない可能性が高い**（推論。2日分の時刻と現在のコードに基づく。修正後の実際の動作は未確認）。
- 実績ページ（`x_results_page`）は複数日の範囲を読み、hit_record に無いレースは結果の API から補う経路を持つ（x_results_common L282）。実績ページへの影響は検証メールより小さいとみる（推論。未確認）。

---

## 5. 未確認

- x_ranking の実行ログで `[DEBUG release_storage] is_available: GITHUB_TOKEN存在=False` が出ているか
- 実績ページ・検証メールの実際の本文
- hit_record.csv が記録されなかった期間。2026-07-06 以降に、Actions 上で hit_record.csv が一度でも Release に保存されたか（現在の Release には無い。実行の一覧は 2026-07-01 以降のものが残っているが、ログ本文は確認していない）
- §2.1 の実行ログ2件以外の実行で、前日の sent が無かったか
- 2026-07-13 の incident の欠落（07-05〜07-07）と本件の関係（同じ原因によるものかは確認していない）
- §4 の実行時刻の並びが、2日以外の日も同じか

---

## 6. 対応（修正の候補：`fix/storage-minimal-20261019` `1f72842`、main 未統合）

**修正で解決する範囲**

- E1：`x_state_sync.py` を追加し、notify_arashi.yml の前後で、前日の sent を Release から取り戻し、当日の sent を保存する。`notify_arashi.py` は変更しない。
- E2：3ステップの env に `GITHUB_TOKEN` を戻す（読み取りのみ。3ステップのスクリプトは Release に書かない）。
- あわせて、Release 保存処理（`x_release_storage.py`）の障害対策を行う。これは `docs/incidents/2026-10-14_buyscore-log-duplicate-and-nonatomic-upload.md` の範囲であり、本書では扱わない。

**修正で解決しない範囲**

- §4 の実行時刻の並びによる、検証メールの前日分の件数。本書では対応を定めない。
- `83069fb` で同じく `GITHUB_TOKEN` が無くなった Update motor history・Generate rankings・Send Weekly Report には戻さない。Generate rankings の `x_venue_stats`（L126）は Release から読むため、戻すと新聞の入力（motor_history）が変わりうる（⑭のスコア比較に影響しうる）。Weekly Report は `download_file` を使っていない。

---

## 7. 既存記録との関係

- `docs/incidents/2026-07-13_hit-record-gap-unrecoverable.md`：「副次的な発見（根本原因）」は「退避対象リストに hit_record.csv が含まれていなかった」と推定している。notify 側の hit_record.csv の Release 保存は commit `7c8390f`（2026-07-06 23:01 JST）で追加されており、同記録の欠落期間（07-05〜07-07）の一部はこの追加より前である。保存の追加より後の期間について、同記録の推定と本書の原因（§2.1）のどちらが当たるかは確認していない。本書は同記録を改訂・置換しない。同記録の対応方針3（新システムの storage 層の要件）にも触れない。
- `docs/incidents/2026-10-14_buyscore-log-duplicate-and-nonatomic-upload.md`：保存の非原子性（N-03）は同記録の範囲で扱う。本書は扱わない。
