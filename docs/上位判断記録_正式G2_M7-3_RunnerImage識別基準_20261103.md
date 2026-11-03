# 上位判断記録：正式G2 M7-3 Runner Image の識別基準と基準の同一性（下書き）

作成日: 2026-11-03 ／ ステータス: **下書き（ユーザー本人の承認待ち。承認前は正式な記録として扱わない）**

## 承認の記録
- 技術方針の判断主体：GPT（ユーザーからの委任に基づく判断。`docs/委任記録_正式G2_判断と進行の委任_20261030.md`）。判断日 2026-11-03。
- ユーザー本人の承認が必要な事項（未承認）：本書による M7-3 の承認済み条件の解釈の確定。GPT は、これをユーザー本人の承認事項として扱うと判断した（2026-11-03）。
- 本書は、正式G2 の開始・baseline の登録・本番切替の許可ではない。

## §1 判断（技術方針）
1. GitHub Actions で実際に使われた Runner Image の識別値は、評価ジョブのログの「Set up job → Runner Image」グループに記録された `Image` と `Version` とする。M7-3 の baseline 照合はこの値で行う。これは GPT の運用上の判断であり、§4 の公式資料の記載に沿う。
2. 評価プロセス内の環境変数 `ImageOS`・`ImageVersion` は、独立した観測値として記録する。照合の基準には使わない。欠落だけでは不適格にせず、欠落した場合は取得できなかったことを記録する。
3. 両者が異なる場合は隠さず、実行の記録に両方の値を残す。
4. ジョブログの取得・解析・検証に失敗した実行は不適格とする。環境変数の値で代わりにしない。
5. 評価プロセスにはトークンを渡さない（C-4）。ジョブログの取得と照合は、集約（`formal_g2_aggregate.yml`）が既存の `actions: read` 権限で行う。
6. 基準（baseline）の同一性は必須とする。評価側は基準の内容の SHA-256 を `run.json` に記録し、実行の終わりにも計り直す。集約側は、自分が使う基準の内容の SHA-256 と照合する。パスが違っても内容の SHA-256 が一致すれば同一とする。未記録・不一致・実行途中の変化・読み込み不能は、実行全体を不適格にする。
7. 基準がない場合、正式な実行は加算しない。
8. 集約の終了コード：照合の失敗（artifact・ジョブログの検証、基準の読み込み、正式な実行の基準の同一性・Runner Image の照合）は 3。評価側の設計どおりの安全停止（CONFIG・PRELOAD など）は、記録と不適格の判定が正しければ 0 とする。
9. baseline の候補は、本書の変更を含むコードで行う新しい調査 run から作る。Q8 の baseline 候補は流用しない。

## §2 環境変数の値を別の観測値として残す理由
- Q8 の同じ run の中で、2つの情報源の値が一致しなかった（§3）。一方だけを残すと、不一致を後から確認できなくなる。
- 環境変数の値は、評価プロセスが実際に受け取った値で、プロセス側の観測記録として意味を持つ。

## §3 Q8 で観測した値
- 対象：調査 run 41401359888（ref `feature/formal-g2-independent`、SHA `ff050e7`、job 125133673866、run_attempt 1）
- ジョブログ L16〜L21（`##[group]Runner Image`）：`Image: ubuntu-24.04`（L17）、`Version: 20261025.353.1`（L18、2026-11-02T06:40:09.9049179Z）
- artifact 13121566522（digest `sha256:7a4534fe70ac9a1769fb76a333e0bb2001808ee9b57e262abb73335585c07627`）の `run.json` `env_snapshot.image`：`ImageOS: ubuntu24`、`ImageVersion: 20261026.354.1`（`os.environ.get` で、2026-11-02T06:40:31Z〜06:40:34Z の間に取得）
- 2つの値が同じ意味の版を指すのか、なぜ異なるのかは、Q8 の証跡からは確定できなかった。
- Q8 のジョブログ：SHA-256 `a14d989c2a4e32cc3baad1e6152bc2a5ad3f0d001452a79982415d7239854bb2`（65,865 bytes、673行）。2026-11-02 と 2026-11-03T12:51Z の2回の取得で同じ値だった。原本は正式な証跡として保存していない。Q8 は調査 run のため、正式G2 の証跡要件には含めない。

## §4 参照先（一次資料）
- https://github.com/actions/runner-images （README の FAQ「What image version is used in my build?」に、「To find out which image version and what software versions are used in a specific build, see `Set up job` (GitHub Actions) ... step log.」とある。§1-1 はこの記載に沿う）
- https://github.com/actions/runner-images/discussions/7661 （GitHub の Discussions。プログラムから版を得る方法として環境変数 `$ImageOS`・`$ImageVersion` が案内されている。ログの値との不一致や、どちらを正とするかには触れていない。§1-1 の根拠ではない）
- https://docs.github.com/en/organizations/managing-organization-settings/configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-organization （artifact とログの保持期間は既定 90 日。公開リポジトリでは 1〜90 日、非公開では 1〜400 日。同じ1つの設定で管理される）
- https://github.blog/changelog/2026-07-17-actions-retention-will-cover-checks-workflow-runs-and-statuses/ （2026-10-01 から、check・workflow run・status も同じ保持期間の設定に従う）

## §5 保持期間の運用条件
- この repo の保持期間の設定値は、2026-11-03 時点で確認できていない（未解決）。Q8 の artifact の期限は、実行から 90 日後の 2027-01-31 だった。
- 評価ジョブのログは、artifact と同じ保持期間の設定に従う。期限を過ぎると、ログも workflow run 自体も削除されうる。
- そのため集約は、評価 run の保持期間内に実行する。ジョブログの原本・SHA-256・取得時刻は集約の出力に保存し、人の承認を得た commit で長期保存する。

## §6 変更しないもの
- Feature Freeze ⑳ の本文、⑳20.3 の例外、設計固定書の文言は変更しない。
- M7 のほかの承認事項（Python のパッチ版、lock の照合、OS 情報、コードの SHA-256、許可リストなど）は変更しない。
- 評価の workflow（`formal_g2_evaluate.yml`）と Q8 の既存 artifact は変更しない。
