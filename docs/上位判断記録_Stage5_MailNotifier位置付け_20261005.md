# 上位判断記録：Stage 5 — 既存 MailNotifier の実装対象としての位置付け

作成日: 2026-10-05 ／ ステータス: **上位判断の記録（判断内容と根拠の記録のみ・実装なし・実送信なし・既存記録の変更なし）**

---

## 1. 対象

`rebuild/notification/notifiers.py` の MailNotifier を、継続基準書 §8 の Stage 5 の実装対象「Gmail送信Notifier」に該当するものとして扱うか。

---

## 2. 結論（ユーザーの判断）

1. 既存の MailNotifier を、Stage 5 の実装対象「Gmail送信Notifier」に該当するものとして扱う。
2. Stage 5 を開始した場合、既存の MailNotifier を Stage 5 の実装対象・成果物として扱う。
3. 本判断は位置付けの判断であり、§7 に挙げる事項を確定しない。

---

## 3. 判断理由

1. 継続基準書 §8 が Stage 5 の実装対象とする「Gmail送信Notifier」に対し、既存の MailNotifier は Gmail SMTP で送信する Legacy `send_email` を呼ぶ Notifier であり、配置も Stage 5 の変更予定ファイルと同じ `notification/` 配下である（F1・F2・F4・F5）。
2. 既存の MailNotifier は、Step5-0 が推奨した既存ラッパー方式（既存の送信関数を import して呼ぶ形）であり、Step5実装計画書の Step5-5 の成果物（`notification/` 配下の Notifier 群）にあたる（F8）。
3. 上記の事実を踏まえ、ユーザーが本判断を行った。Step 体系と Stage 体系の対応は既存資料に記録がなく（F9）、本判断は両体系の対応を一般に定めるものではない。

---

## 4. 確認された事実

| # | 事実 | 出典 |
|---|---|---|
| F1 | 継続基準書 §8 の Stage 5 には、目的として「Gmailへの実送信を行うNotifierを実装する」、実装対象として「Gmail送信Notifier。Shadow運用中は`NullNotifier`のまま実送信禁止を維持する設計とする」と記載されている。本書は、この記載を引用するのみであり、記載どおりの構成が実装上満たされているかを判断しない | 継続基準書 §8 |
| F2 | 継続基準書 §8 は、Stage 5 の変更予定ファイルを「`notification/`配下（新規）」としている | 継続基準書 §8 |
| F3 | 継続基準書 §5 の C6 の現状欄は「未着手。`shadow/notifier.py`は`NullNotifier`のみ存在。実送信Notifierは存在しない」と記載している。この記載は `314774c`（2026-09-08）で追加され、§5 の見出しは「実コード確認済み・現状反映」である | 継続基準書 §5、`314774c` |
| F4 | `rebuild/notification/notifiers.py` の MailNotifier は `ad25a82`（2026-07-25）で追加され、その時点で Legacy `send_email` を呼ぶラッパーだった | `ad25a82` |
| F5 | Legacy `send_email` は Gmail SMTP（smtp.gmail.com:587、TLS）で送信する | `notify_arashi.py` |
| F6 | MailNotifier は `5efc033`（2026-09-13）で `build_production_bundle()` の通知構成に配線され、`0c726e9`（2026-09-20）で per-race の `request.body` を送信関数へ渡すよう変更された | `5efc033`、`0c726e9` |
| F7 | MailNotifier から Gmail へ実際に送信した記録はない。C5 の実データ確認は NullNotifier で行われ、MailNotifier を通っていない | `docs/上位判断記録_Stage4_C5_実データ確認_20261005.md` §4 |
| F8 | Step5実装計画書は、Step5-5 の成果物を「`notification/`配下Notifier群」とし、mail を含む送信クライアントを DI で注入する方針としている。Step5-0 は既存ラッパー方式を推奨している | Step5実装計画書 §Step5-5、Step5-0 |
| F9 | Step 体系と Stage 体系の対応は記録されていない | `docs/上位判断記録_Stage3_20260927.md` §3 |
| F10 | 本書の作成時点で、MailNotifier の per-race の `request.body` が送信関数へ渡ることを確認する専用テストはない。Shadow 側の実際の `build_bundle()` の NullNotifier 登録を直接確認する単体テストはない | `rebuild/tests/unit/`（`9862585` 時点） |

次の事項は、それぞれ別の事項として扱う。

- MailNotifier が存在すること（F4：事実）
- 既存の MailNotifier を Stage 5 の実装対象「Gmail送信Notifier」として扱うこと（§2：本書のユーザーの判断）
- Stage 5 の実装完了、C6 の達成、Gmail への実送信の確認、Stage 5 の完了、Stage 6 の開始（§7：本書では決めない）

---

## 5. 過去資料との関係

1. 継続基準書 §8 の「`notification/`配下（新規）」（F2）と、§5 の C6 の現状欄「実送信Notifierは存在しない」（F3）の記載は、そのまま残す。本書はこれらを訂正・改訂しない。
2. 現在のコードには、F3 の記載より前から MailNotifier が存在している（F4）。
3. 本書は、上記 1 と 2 の差を事実として記録する。差が生じた原因は記録の対象としない。
4. Stage 5 の実装対象としての MailNotifier の位置付けは、本書による。

---

## 6. Stage 5 の開始条件との関係

1. Stage 5 の開始条件は、継続基準書 §8 のとおり Stage 4 の完了である。本書はこれを変更しない。
2. 本書は、Stage 5 を開始してよいとする新しい条件を定めない。Stage 5 の開始を判断しない。
3. 本書が確定するのは、Stage 5 を開始した場合の既存 MailNotifier の位置付けのみである。

---

## 7. 本判断が意味しないこと

本判断によって、次の事項は確定しない。

- Stage 5 の実装が完了したこと
- Stage 5 の実装作業が不要であること
- Stage 5 が完了したこと
- C6 を達成したこと
- Gmail へ実際に送信できることが確認されたこと
- Stage 6 へ進めること
- G2・G3 が解消したこと、GO 判定に進めること
- Phase 1 を開始できること

---

## 8. 今後の C6・Stage 5 完了判定との関係

1. C6 の達成・未達は、`docs/上位判断記録_Stage5_C6_判定方法保留_20261005.md` のとおり現時点では判定しない。C6 の判定方法は、引き続き別途正式判断の対象とする。
2. Stage 5 の完了判定を行う場合、本書により、判定の対象となる実装は既存の MailNotifier となる。
3. F7・F10 の事実（実送信の記録がないこと、テストで確認されていない経路があること）は、本書では評価しない。C6 の判定方法・Stage 5 の完了判定において、どう扱うかは決めていない。

---

## 9. 判断後の状態

| 項目 | 状態 |
|---|---|
| 本書の論点（既存 MailNotifier の位置付け） | CLOSED（USER DECISION により確定） |
| C6 の達否 | 未判定 |
| C6 の判定方法 | USER DECISION（別途正式判断） |
| Stage 5 の開始 | 未判断 |
| Stage 5 の完了 | 未判定 |
| Stage 6 の開始 | 未判定 |
| G2・G3 | 未解決 |

---

## 10. 変更していないもの

- Phase0.5 設計固定書 v1.1.8
- 継続基準書（§5 の C6 の現状欄、§8 の Stage 5 の記載を含む）
- Step5-0、Step5実装計画書、Step6-0
- 既存の上位判断記録の各件
- MailNotifier の実装、本番 bundle の構成、Shadow の NullNotifier 強制
- 新しい完了条件、判定基準、Go/No-Go 条件、工程番号

本判断に伴う実装の変更、実送信、Shadow の実行は行っていない。

---

## 11. 次に必要な判断・調査

- C6 の判定方法（USER DECISION。`docs/上位判断記録_Stage5_C6_判定方法保留_20261005.md` による）
- Stage 5 の開始の判断

---

本判断により、既存の MailNotifier を Stage 5 の対象として扱うことは確定した。C6 の達成、Stage 5 の完了、Gmail への実送信の確認は確定していない。
