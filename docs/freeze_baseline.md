# Feature Freeze ベースライン記録

根拠: Phase 0.5 設計固定書 v1.1 ⑳20.4（凍結の実施手続き）

| 項目 | 値 |
|---|---|
| 凍結宣言日 | 2026-07-13 |
| gitタグ | freeze-v1-baseline |
| 凍結期間 | 2026-07-13 〜 Phase 5 完了時 |
| 比較基準 | 本タグ時点の現行システム（notify_arashi系一式） |

## 凍結時点の設定ファイルハッシュ（SHA-256）

| ファイル | SHA-256 |
|---|---|
| asahi_config.json | 6a7862b8dc36006854d557fa8e2bfd12823433a0c51c5f092bb7f357b57520b3 |
| buyscore_config.json | da6a4edaf8220aa52ed3e4577c23ff443509512a971d0ba322079eb4c4cd6f1d |

検証方法（Windows）:
```
certutil -hashfile asahi_config.json SHA256
certutil -hashfile buyscore_config.json SHA256
```
上表と一致しない場合、凍結基準と手元の設定が乖離している。作業を停止し原因を確認すること（実装ルール10）。

## 凍結に伴う変更（⑳20.2-4 buyscore_tuner 自動反映の停止）

- x_ranking.yml 226行目: `x_buyscore_tuner.py` に `--dry-run` を付与
- 効果: チューニング分析・提案ログは毎晩継続出力、buyscore_config.json への保存のみ停止
- 解除条件: Phase 5 完了（新システム切替後、tuner後継 ml/tuner.py 側で再開）

## 凍結期間中の禁止事項（要約・全文は設計書⑳20.2）

1. 新機能追加の禁止
2. アルゴリズム変更の禁止
3. スコア変更の禁止（config閾値・重みの変更を含む）
4. tuner自動反映の停止（本記録で実施済み）
5. スキーマ変更の禁止
6. リファクタリング・改名・移動の禁止

例外（障害修正・セキュリティ修正・観測ログ追加・データ復旧）は設計書⑳20.3の条件に従い、実施時は docs/incidents/ へ記録する。
