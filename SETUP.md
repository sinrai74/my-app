# ボートレース荒れ検知（notify_arashi）セットアップ

全ボートレース場の出走表をもとに、**1号艇以外が1着になりそうなレース** を評価し、
予想・購入判定のデータを蓄積します。

> 【2026-10 訂正】旧版は LINE Notify による通知、展示・気象を使った配点、閾値 3.0 を前提に
> 書かれていたが、commit `693e78d` 時点の実装と一致していなかったため、コードで確認できた事実に
> 合わせて書き直した。仕様・動作・閾値の変更ではない。
>
> 通知について（commit `693e78d` のコードで確認した状態）:
> - `notify_arashi.py` の中で `send_notification()`（LINE とメール）を呼ぶ行は2か所あり、どちらもコメントアウトされている。
> - レースごとの通知箇所は通知関数を呼ばず、「通知スキップ（蓄積のみ）」のログを出す
>   （コメント「メール通知は停止（転がし専用エンジンに移行）」）。
> - `send_email()`・`send_line()` は Rebuild（`rebuild/notification/notifiers.py`）から呼べる形に実装されている。
>   その実運用への切替は GO 判定後とされている（`rebuild/actions/production_entrypoint.py` のコメント）。
>   commit `693e78d` の `.github/workflows/` には、Rebuild の本番入口（`production_entrypoint`）を起動する
>   ワークフローはない（`shadow_run.yml` は `USE_REBUILD_PIPELINE: "False"`）。
> - `send_line()` は、環境変数 `LINE_NOTIFY_TOKEN` が未設定なら、debug ログを出して送信せずに `False` を返す。

---

## ファイル構成（本機能に関係するもの）

```
my-app/
├── notify_arashi.py                     ← 本体
├── x_asahi_scoring.py                   ← 荒れスコア（calculate_upset_score_v2）・危険度の算出
├── x_buyscore.py                        ← 購入判定（BuyScore）
├── x_release_storage.py                 ← GitHub Releases（data-store）へのファイル保存・取得
├── csv_schema.py / migration.py         ← hit_record.csv のスキーマ管理（現在 v4・44列）
├── model_all.pkl                        ← 1着確率などの学習済みモデル
├── asahi_config.json / buyscore_config.json
└── .github/workflows/notify_arashi.yml  ← GitHub Actions
```

---

## GitHub Secrets

Settings → Secrets and variables → Actions → New repository secret

| Name | 用途 |
|------|------|
| `GMAIL_ADDRESS` | `run()` の冒頭で、`GMAIL_ADDRESS` と `GMAIL_APP_PASS` のどちらかが未設定なら、エラーを出して終了する。通知が停止中でも必要 |
| `GMAIL_APP_PASS` | 同上 |

`GITHUB_TOKEN` は Secrets への登録は不要。ワークフローが `secrets.GITHUB_TOKEN` を
「Run upset detector」ステップの環境変数として渡し、`x_release_storage.py` が
`GITHUB_TOKEN` と `GITHUB_REPOSITORY` の環境変数を使って Releases にアクセスする。

---

## 実行方法

### notify_arashi.py を実行したときの前提と動作（コードで確認した範囲）

- 環境変数 `GMAIL_ADDRESS`・`GMAIL_APP_PASS` が未設定だと、`run()` の冒頭でエラー終了する。
- 実行すると、出走表・直前情報・3連単オッズなどを外部から取得し、作業ディレクトリにファイルを書き込む。
  環境変数 `GITHUB_TOKEN` / `GITHUB_REPOSITORY` が無い環境では、Releases への保存・取得はスキップされる。
- `--date`（`-d`）：対象日を `YYYYMMDD` で指定する。指定しない場合は JST の当日。
- 環境変数 `SKIP_TIME_FILTER=1`：次の3か所の絞り込みを無効にする。
  - `run()` の時間帯フィルタ（下記）
  - `_run_main()` の「締切前のレースだけに絞る」処理
  - `_run_main()` のオッズを先に並列取得する対象から、締切済みのレースを除く処理

### hit_record.csv のスキーマ状態の確認

`python migration.py --check` は、`check_schema()` の結果（ファイルの有無・現在のバージョン・期待バージョン・
変換の要否）を表示して終了し、変換処理（`run_migration()`）は呼ばない。
`check_schema()` の内部の動作は、本書の作成時には確認していない。

### GitHub Actions

- **定期実行**：`.github/workflows/notify_arashi.yml` の cron（UTC 23:00〜13:00 ＝ JST 8:00〜22:00）。
  引数は付けずに `python notify_arashi.py` を実行する。
- **手動実行**：Actions タブ → `workflow_dispatch`。入力欄は `race_date` / `threshold` / `skip_filter`。
  - `race_date` が空でなければ `--date`、`threshold` が空でなければ `--threshold` を付けて実行する。
  - `skip_filter` は環境変数 `SKIP_TIME_FILTER` として渡される。

---

## 実行タイミング

- **ワークフローの cron**：UTC 23:00〜13:00（JST 8:00〜22:00）の範囲で起動する。
- **コード側の時間帯フィルタ**：`run()` は JST 8:20〜22:45 以外の起動を「時間帯外のためスキップ」で終える
  （`SKIP_TIME_FILTER=1` のときは行わない）。
- cron は起動の予定であり、実際の起動時刻・回数は GitHub 側の状況による。

---

## 判定の流れ（概要）

1. 出走表（BoatraceOpenAPI programs v2）を取得する。
2. `x_asahi_scoring.calculate_upset_score_v2()` で荒れスコアを算出する。
3. `model_all.pkl` から 1着確率が得られ、1号艇以外の最大確率が1号艇の確率を上回るとき、
   `min(その確率 × 8.0, 4.0)` を荒れスコアに加える。
4. 下記の閾値で判定する。

この前後にも、場・条件による見送り（`_should_skip_by_condition()`）、レース品質による見送り
（`_check_race_quality()`）などの処理があり、上は全体の一部である。買い目候補の作り方は本書では扱わない。
詳細な配点は `scoring_spec.md` を参照。

---

## 閾値（【2026-10 訂正】）

- **場別の閾値**：`VENUE_THRESHOLDS` に、場番号 1〜24 の24場すべての値（4.8〜6.0）が設定されている。
- **一般の既定値**：`UPSET_SCORE_THRESHOLD = 6.0`。判定では
  `VENUE_THRESHOLDS.get(venue_num, UPSET_SCORE_THRESHOLD)` で取得するため、この値が使われるのは
  `VENUE_THRESHOLDS` にない場番号のときだけである。
- **`--threshold`（`-t`）**：`UPSET_SCORE_THRESHOLD` を書き換え、「▶ 開始」のログに表示される。
  24場すべてに場別の閾値があるため、現状では 1〜24 の場の判定には実質的に使われない。
- **展示データの補正**：展示データを使わない設定（`has_exhibition = False`）のため、
  判定に使う閾値には常に +1.5 が加わる。
- **判定式**：
  - 買い目候補（EV のシグナル）がない場合：荒れスコア ＜ 閾値 ならスキップ。
  - 買い目候補がある場合：荒れスコア ＜ 閾値 × 0.6 ならスキップ（閾値は +1.5 を加えた後の値）。
- **手動実行の既定値**：ワークフローの入力欄 `threshold` の既定値は `"4.0"`。入力欄の説明文は
  「デフォルト: 3.0」となっており、既定値と一致していない（ワークフローは変更していない）。
- **定期実行**：`--threshold` を付けないため、ログには既定の 6.0 が表示されるはずだが、
  定期実行のログでは確認していない。

---

## モデルと再学習

`model_all.pkl` は gzip で圧縮した pickle で、`notify_arashi._load_ml_model()` が読み込む。
中身の `models` には `retrain.py` が `win` / `top3` / `rank` / `payout` を格納する。

```python
import gzip, pickle
with gzip.open("model_all.pkl", "rb") as f:
    payload = pickle.load(f)
win_model = payload["models"]["win"]
```

再学習は `.github/workflows/retrain.yml`（cron: UTC 日曜 18:00 ＝ JST 月曜 3:00）から `retrain.py` で行う。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS が設定されていません` で終了 | Secrets 未設定 | 「GitHub Secrets」の2つを登録する |
| `[スキーマ] hit_record.csv のスキーマ状態が安全に判定できません` で終了 | hit_record.csv のヘッダーが既知のスキーマと一致しない | `python migration.py --check` で状態を確認する |
| `時間帯外のためスキップ` で終了 | JST 8:20〜22:45 以外の起動 | 正常な動作。確認のために実行する場合は `SKIP_TIME_FILTER=1` |
| `出走表が取得できませんでした（開催なし or API 障害）` | 開催なし or API 障害 | 翌日再確認 |
| 直前情報なし | レース前（未公開） | 15分後に再実行で取得可能 |
