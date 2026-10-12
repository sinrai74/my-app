# incident：本番で ML モデルを読み込めない（scikit-learn の版が固定されていない）

記録日: 2026-10-12 ／ ステータス: **修正前の記録**（設計固定書 v1.1.8 ⑳20.3、正式判断 D4）

区分: **確認事実**＝コード・ログで直接確認した ／ **未確認**

---

## 1. 事象（確認事実）

| # | 内容 | 根拠 |
|---|---|---|
| E1 | 本番の notify の実行ログで、ML モデルの読み込みに失敗していることを確認した | 2026-10-12、ユーザーが提供した GitHub Actions のログ |
| E2 | `.github/workflows/notify_arashi.yml` L51 は `pip install requests beautifulsoup4 pandas numpy scikit-learn lightgbm` で、scikit-learn・LightGBM の版を指定していない | blob `90e8e27198249e3b0e7d6e1df951d15f069ff4cb`（commit `ccce216` 時点） |
| E3 | モデルを作る `retrain.yml` L41 は `scikit-learn==1.5.2`・`lightgbm==4.5.0` で固定している。`x_korogashi.yml` L42・`x_manshuu_watch.yml` L40・`x_ranking.yml` L58 は `scikit-learn==1.5.2`、`requirements-ml.txt` は `scikit-learn==1.5.2`・`lightgbm==4.5.0` | 各ファイル |
| E4 | scikit-learn 1.5.2 の環境では、現行のモデル（blob `2dad1b9…`）と、D3 修正後の再学習モデル（blob `38449d31…`）のどちらも、`notify_arashi._load_ml_model()` が True を返す | 手元での確認 |

---

## 2. 影響（確認事実）

- 読み込みに失敗している間、`_predict_win_prob`（L288）は空の結果を返す。`_run_main` は `if ml_probs:` の分岐（L1900・L1909・L2477・L4245・L4262 ほか）に入らず、EV 評価・BuyScore・購入判定・購入ログの保存は動かない。
- メール送信は、本障害とは関係なく停止している。`send_notification` を呼ぶ箇所は L4076・L5309 のコメントの中だけで、`send_email`・`send_line` は `send_notification` の中（L3100–3101）からしか呼ばれない。

---

## 3. オッズ解析障害との関係（D4、F6・F7）

- ML の読み込み障害とオッズ解析障害は別の障害である（D1）。ML を読み込めるようになっても、オッズ解析の正確性は保証されない。
- 読み込みが復旧すると、オッズ解析の結果を使う後続処理（EV 評価・BuyScore・購入判定・記録）が再開する（F7）。
- オッズ解析は commit `ccce216` で修正された。実際のページ3件で全組番がページの値と一致し、欠場艇を含む組番は0件だった（`docs/上位監査記録_J1_D3_修正検証_20261012.md` §4）。2艇以上の欠場、レース中止、発売前の実際のページは未確認である。
- オッズ解析障害の発生期間と損益への影響は未確定である（`docs/上位判断記録_J1-a_オッズ解析障害分類_20261012.md` §6）。

---

## 4. 読み込みが復旧した場合に動く Release への書き込み（確認事実）

- `x_buyscore.save_buyscore_log`（L595–640）は、Release の `buyscore_log.jsonl` に書き込む。`x_release_storage.upload_file` は既存の asset を `.bak` に退避してから置き換える。
- `notify_arashi._check_yesterday_results`（L4972〜）は、前日の `sent_{日付}.txt` に記録があると、Release の `hit_record.csv` に書き込む（L5215）。GitHub Actions では `sent_*.txt` は `.gitignore` の対象で（L32）、workflow に保存の手順がないため、翌日の実行時には存在しないとみられる（コードからの推論。実行での確認はしていない）。
- `x_release_storage` は、対象の Release がない場合に新しく作る（`_get_or_create_release`、L86–130）。

---

## 5. 修正方針（B1）

`.github/workflows/notify_arashi.yml` L51 だけを変更し、`"scikit-learn==1.5.2" "lightgbm==4.5.0"` に固定する。`notify_arashi.py` は変更しない。適用の可否と条件は、別の上位判断記録で判断する。

---

## 6. 未確認

- 障害が始まった時期（scikit-learn の新しい版の公開時期との関係）
- 最新版の scikit-learn での読み込みの失敗を、手元では再現していない（E1 は本番のログによる）
- 購入ログの重複：`save_buyscore_log` は同じレースの記録の有無を確認しない。定期実行ごとに同じレースが重ねて記録されるかは未確認
- `upload_file` は原子的でなく、同時実行を防いでいない
