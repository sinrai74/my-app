# ボートレース荒れ検知 LINE 通知システム

15分ごとに全ボートレース場の最新データを取得し、
**1号艇以外が1着になりそうなレース** を LINE Notify で自動通知します。

---

## ファイル構成

```
your-repo/
├── notify_arashi.py                     ← 本体スクリプト
├── .github/
│   └── workflows/
│       └── notify_arashi.yml            ← GitHub Actions ワークフロー
└── SETUP.md                             ← このファイル
```

---

## セットアップ手順

### 1. LINE Notify トークンを取得する

1. [LINE Notify](https://notify-bot.line.me/ja/) にアクセス
2. 「マイページ」→「トークンを発行する」
3. 通知先のトーク名を選択してトークンを発行
4. 発行されたトークン文字列をコピー（一度しか表示されません）

---

### 2. GitHub Secrets にトークンを登録する

1. GitHub リポジトリページの **Settings** → **Secrets and variables** → **Actions**
2. **「New repository secret」** をクリック
3. 以下を入力して保存：

| Name | Value |
|------|-------|
| `LINE_NOTIFY_TOKEN` | 手順1で取得したトークン |

---

### 3. リポジトリにファイルをプッシュする

```bash
git add notify_arashi.py .github/workflows/notify_arashi.yml
git commit -m "Add boat race upset detector"
git push origin main
```

プッシュ後、GitHub Actions タブで自動実行が開始されます。

---

## 通知メッセージの例

```
【荒れ検知】福岡 5R
🌊 風: 6.0m 向 / 波: 15cm / 天候: 晴
🚤 1号艇 山田 太郎: 展示 6.85秒 / ST 0.18
⚠️  1号艇危険度: 🟠 高 (スコア: 5.50)
🎯 狙い: 3-2-全
```

---

## スコアリングロジック

| 判定項目 | 配点 | 条件 |
|---------|------|------|
| 展示タイム劣勢 | +2.0 | 1号艇が3位以下 |
| 展示タイム差大 | +1.5 | 最速との差 ≥ 0.05秒 |
| 展示ST遅れ | +2.0 | 1号艇 ST ≥ 0.18 |
| 他艇ST大幅速い | +1.0 | 2艇以上が0.03以上速い |
| 強風 | +1.5 | 風速 ≥ 5m/s |
| 向かい風 | +1.0 | 風向 = 向かい風 |
| 高波 | +1.0〜2.0 | 波高 ≥ 10cm |
| 勝率劣勢 | +1.5 | 1号艇勝率 < 他艇平均×85% |
| モーター劣勢 | +1.0 | 1号艇モーター率 < 他艇平均×90% |

**閾値: 合計スコア ≥ 3.0 で通知**（`UPSET_SCORE_THRESHOLD` で変更可能）

---

## 実行タイミング

- **自動**: JST 8:00〜22:00 の間、15分ごと
- **手動**: GitHub Actions タブ → `workflow_dispatch` から任意の日付で実行可能

---

## 将来の機械学習モデル組み込み

`notify_arashi.py` の `MODEL_SCORER` 変数に predict 関数をセットするだけで、
既存のルールベーススコアに ML スコアを上乗せできます。

```python
# notify_arashi.py の先頭付近に追加するだけ
import pickle
payload = pickle.load(open("model_all.pkl", "rb"))
MODEL_SCORER = payload["models"]["win"].predict_proba
```

`score_by_model()` 関数内に特徴量変換ロジックを実装してください。
`features.py` の `FEATURE_COLS` や既存の特徴量エンジニアリング関数がそのまま使えます。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| 通知が来ない | 荒れスコア閾値以上のレースなし | 正常動作。閾値を下げて試す |
| `LINE_NOTIFY_TOKEN` エラー | Secrets 未設定 | セットアップ手順2を実施 |
| 出走表取得失敗 | 開催なし or API 障害 | 翌日再確認 |
| 直前情報なし | レース前（未公開） | 15分後に再実行で取得可能 |
