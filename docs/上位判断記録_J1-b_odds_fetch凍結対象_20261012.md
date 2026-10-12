# 上位判断記録：J1-b — `odds_fetch.py` を Feature Freeze の対象として扱う

作成日: 2026-10-12 ／ ステータス: **上位判断の記録（ユーザー判断の記録・実装なし・既存記録の変更なし）**

---

## 1. 位置付け

`docs/incidents/2026-10-09_odds-parser-combo-mismatch.md` に記録された3連単オッズ解析の障害（以下「オッズ解析障害」）の修正に先立ち、修正対象 `odds_fetch.py` を Feature Freeze（設計固定書 v1.1.8 ⑳）の対象として扱うかを判断する。

- 基準 commit：`dedef3a`（origin/main。`298ca62` 以降はデータの自動 commit のみで、コード・ワークフロー・docs の差分はない）
- 「J1-b」はチャット上の整理番号であり、本書の題名に用いるだけで、新しい工程番号を定義しない。
- 本書は `docs/上位判断記録_J1-a_オッズ解析障害分類_20261012.md` の前提となる。

---

## 2. FACT

| # | 事実 | 出典 |
|---|---|---|
| F1 | ⑳20.1 は、凍結の対象を「現行システム（notify_arashi系一式）」とする。対象ファイルの一覧はない | `docs/Phase0_5_設計固定書.md` ⑳20.1 |
| F2 | `docs/freeze_baseline.md` が記録しているのは、タグ `freeze-v1-baseline` と、設定ファイル2件（asahi_config.json・buyscore_config.json）のハッシュだけである | `docs/freeze_baseline.md` |
| F3 | `odds_fetch.py` を凍結の対象として名指しした正式資料はない | docs/ 全体の確認 |
| F4 | `odds_fetch.py` は `notify_arashi.py` から import され（L3411・L3513）、現行システムの稼働に使われている | `notify_arashi.py` |
| F5 | `odds_fetch.py` の内容は `freeze-v1-baseline` 時点と同一である（blob `76b1cb4`） | `git rev-parse freeze-v1-baseline:odds_fetch.py` |
| F6 | `odds_fetch.py` は Rebuild（`rebuild/actions/shadow_entrypoint.py` L173・L201、およびそれを再利用する `rebuild/actions/production_entrypoint.py`）からも呼ばれる | 同ファイル |

---

## 3. 判断（ユーザー判断、2026-10-12）

| # | 判断 | 内容 |
|---|---|---|
| J1-b | `odds_fetch.py` を Feature Freeze の対象に含める | 本判断は ⑳20.1 の文言から確定する解釈ではなく、F4・F5 を理由として採用する**運用上の判断**である。したがって、`odds_fetch.py` の変更は ⑳20.2 の禁止事項と ⑳20.3 の例外手続きに従う |

### 判断の範囲

- 本判断は `odds_fetch.py` の扱いに限る。「notify_arashi系一式」の範囲を一般的に定めるものではなく、他のファイルの凍結対象への該当性を判断しない。
- 本判断は、オッズ解析障害の分類、修正方法、実装、適用を承認しない（分類は J1-a の記録による）。

---

## 4. 判断しない事項

- 「notify_arashi系一式」の範囲の定義
- `retrain.py`・`parsers.py` その他のファイルが凍結対象に当たるか
- Rebuild のコードに対する ⑳ の適用

---

## 5. 既存記録との関係

本書は、設計固定書 v1.1.8、`docs/freeze_baseline.md`、継続基準書、`docs/上位判断記録_整合性レビュー対応方針_20261009.md`（D1〜D7・A1〜A4）、既存の incident を改訂・置換しない。新しい完了条件・判定基準・Go/No-Go 条件・工程番号は定義しない。
