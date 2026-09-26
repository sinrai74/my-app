# 競艇AIプラットフォーム Phase 0.5 設計固定書（Architecture Freeze）

制定日: 2026-07-13 ／ 版: **1.1.8**
状態: **確定（Frozen）**
本書は競艇AIシステム唯一の設計基準である。Phase 1以降、本書の変更は「本書⑯のバージョン管理手続き」を経ない限り認めない。

---

# ① システム理念

## 1.1 定義

本システムは **競艇AIプラットフォーム** である。

AI新聞を作るシステムではない。危険艇速報を送るシステムでもない。それらはすべて、単一のAIエンジンが生成した**評価データ**から派生する成果物にすぎない。

## 1.2 中心命題

**「1回の評価、無限の成果物」（Evaluate Once, Publish Everywhere）**

1レースにつきAIエンジンは1回だけ評価を行い、その結果を `RaceEvaluation` として保存する。新聞・ランキング・実績ページ・メール・X投稿・将来のWeb/APIは、すべてこの保存済み評価データを読むだけで生成される。

## 1.3 三原則

1. **評価は一度だけ**: 同一レースのスコアを複数箇所で再計算しない。表示層・配信層でのスコア計算は禁止。
2. **データが契約**: レイヤー間の接続はデータモデル（③）のみで行う。関数の内部実装への依存を禁止する。
3. **書けなければ評価していない**: 評価結果は保存されて初めて存在する。「計算したが保存されない」経路を作らない（danger_score_v3欠落事故の再発防止原則）。

## 1.4 このシステムが提供する価値

- 利用者へ: 毎朝の判断材料（新聞・ランキング・警報）と、隠さない実績公開（透明性）
- 運営者へ: 自動運転（GitHub Actions）・自己改善（検証→チューニング→再学習）・監査可能性（全評価が記録に残る）

---

# ② システム全体アーキテクチャ

## 2.1 レイヤー構成図

```
┌──────────────────────────────────────────────────────┐
│ L0 外部データ層 (data/)                                │
│   BoatraceOpenAPI / beforeinfo / FANファイル /         │
│   Kファイル / オッズ                                    │
└──────────────┬───────────────────────────────────────┘
               ▼ Race, RaceEntry, OddsSnapshot
┌──────────────────────────────────────────────────────┐
│ L1 特徴量層 (features/)                                │
│   MotorHistory / VenueStatistics / LocalCourseStats    │
│   → FeatureSet の構築（AIエンジンへの唯一の入力）        │
└──────────────┬───────────────────────────────────────┘
               ▼ FeatureSet
┌──────────────────────────────────────────────────────┐
│ L2 AIエンジン層 (core/)  ※純粋関数・副作用ゼロ          │
│   EvaluationEngine（Ver4/Ver5/MLHybrid差替可能）        │
│   評価 / danger / upset / hot / awakening /            │
│   買い目生成 / BuyScore購入判定                         │
└──────────────┬───────────────────────────────────────┘
               ▼ RaceEvaluation, Prediction, BuyDecision
┌──────────────────────────────────────────────────────┐
│ L3 保存層 (storage/)                                   │
│   CSV / JSON / GitHub Releases / Cache / 冪等性管理    │
│   ※書込＝Releases退避＝commitを不可分に実行            │
└──────────────┬───────────────────────────────────────┘
               ▼ 保存済みデータのみ
┌──────────────────────────────────────────────────────┐
│ L4 表示層 (output/)  ※判定禁止・読み取り専用            │
│   AI新聞HTML / ランキング画像 / 実績ページ /            │
│   メール本文 / X投稿文                                  │
└──────────────┬───────────────────────────────────────┘
               ▼ 成果物ファイル
┌──────────────────────────────────────────────────────┐
│ L5 配信層 (services/)                                  │
│   メール送信 / GitHub Pages反映 / (将来)X投稿           │
│   ※冪等性チェックの唯一の実行地点                       │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ L6 学習層 (ml/)  ※夜間・週次の逆流パイプライン          │
│   HitRecord → 検証 → LearningData → 再学習 →           │
│   モデル保存 → チューニング反映                          │
└──────────────────────────────────────────────────────┘

※全レイヤー（L0〜L6）は実行結果を SystemMetrics（③3.16）として
  storage へ記録する。これはデータ依存ではなく計測の横断関心事である。
```

## 2.2 各レイヤーの責務（確定）

| レイヤー | 責務 | 禁止事項 |
|---|---|---|
| L0 data | 外部ソースからの取得・パース・リトライ・データモデルへの正規化 | スコア計算、保存判断 |
| L1 features | 履歴データから特徴量を構築し **FeatureSet** に集約する | 外部API直接呼出（L0経由必須）、購入判定 |
| L2 core | 評価・スコア・判定・購入判断。入力=Race+FeatureSet、出力=データモデル | ファイルI/O、ネットワーク、環境変数参照、日時取得（引数で受け取る） |
| L3 storage | 永続化・整合性・バックアップ・冪等性記録・メトリクス記録の管理 | データの加工・解釈 |
| L4 output | 保存済みデータ→HTML/画像/テキストへの変換 | API呼出、スコア再計算、保存層への書込（成果物ファイル出力を除く） |
| L5 services | 送信・公開・通知。送信前の冪等性チェック | 評価、集計、本文の組み立て |
| L6 ml | 特徴量生成→学習→モデル保存→評価→反映 | 本番評価パスへの直接介入（モデルはRelease経由でのみ受け渡す） |

## 2.3 依存方向（確定）

依存は**上から下（L0→L5）へ一方向**。逆流はL6（学習）のみが持ち、L6はL3の保存済みデータだけを入力とする。

- core は data / storage / output / services を import してはならない
- output / services は core を import してはならない（データモデル定義のみ共有可）
- 全レイヤーが共有できるのは `models/`（データモデル定義）と `config/` のみ

---

# ③ データモデル固定（最重要）

全モデルは `models/` に dataclass として定義する。JSON・CSV・HTMLはすべてこのモデルから生成する。モデル定義が唯一の真実であり、CSVヘッダーやJSONキーを手書きで増やすことを禁止する。

型表記: str / int / float / bool / date(YYYYMMDD文字列) / json(シリアライズ済み文字列)。「必」=必須、「任」=Optional。

## 3.1 Race（レース基本情報）

| 項目 | 型 | 必須 | 説明 |
|---|---|---|---|
| race_date | date | 必 | 開催日 |
| venue_num | int | 必 | 場コード 1-24 |
| venue_name | str | 必 | 場名 |
| race_number | int | 必 | 1-12 |
| close_time | str | 必 | 締切時刻 HH:MM |
| is_night | bool | 必 | ナイター |
| grade | str | 任 | SG/G1/G2/G3/一般 |
| entries | list[RaceEntry] | 必 | 6艇 |
| weather | Weather | 任 | 荒れ指数（upset_score）算出の評価入力として使用する。§3.4 RaceEvaluationには保持せず、EvaluationEngine.evaluate()への引数として渡す（§5.2） |

生成者: data/openapi_client ／ 利用者: features(FeatureSet構築), core ／ 保存先: 保存しない（評価結果のみ保存）

## 3.2 RaceEntry（出走艇1艇）

| 項目 | 型 | 必須 |
|---|---|---|
| lane | int | 必 |
| racer_no | str | 必 |
| racer_name | str | 必 |
| racer_class | str | 必 |
| branch / hometown | str | 任 |
| win_rate / place_rate | float | 必 |
| motor_no | int | 必 |
| motor_rate2 | float | 必 |
| boat_no / boat_rate2 | int/float | 任 |
| avg_st | float | 必 |
| course_stats | CourseStats | 任 | FANファイル由来（コース別複勝率・ST・F/L回数） |

生成者: data/openapi_client + data/fan_file ／ 利用者: features, core ／ 保存先: 保存しない

## 3.3 FeatureSet（AIエンジンへの唯一の入力・新設）

**位置付け: AIエンジン（L2）が受け取る特徴量はこのモデルに限る。** coreがRaceEntryの生値を直接読んで独自加工することを禁止し、「どの特徴量で判定したか」を1箇所に固定する。boat_features のキー集合は RaceEvaluation.features（3.4）の保存キーと1:1対応させる。

| 項目 | 型 | 必須 | 説明 |
|---|---|---|---|
| eval_id | str | 必 | 対象レース `{date}_{venue:02d}_{race:02d}` |
| feature_schema_version | int | 必 | キー集合の版 |
| built_at | str | 必 | ISO8601 JST |
| boat_features | dict[int, dict[str, float]] | 必 | 艇番→特徴量。キー: win_rate, place_rate, motor_rate2, avg_st, racer_class_score, course_st_1c, course_rank_1c, course_f_rate_1c, course_l_rate_1c, course_rentai2_1c, course_sample_confidence, ability_trend, motor_recent_score, exhibition系(朝刊ポリシーによりnull固定) |
| race_features | dict[str, float] | 必 | レース単位。キー: venue_factor, venue_water_type_code, field_strength, class_gap |
| local_features | dict[int, dict[str, float]] | 任 | 当地コース特徴量（Phase 3接続まではnull） |
| missing_keys | list[str] | 必 | 計算不能だった特徴量名（null と未計算の区別。空リスト可） |

- 生成者: features/feature_builder（L1の唯一の出口）
- 利用者: core/EvaluationEngine（唯一の消費者）、ml/training_export（学習時の再現用）
- **保存方法: FeatureSet自体は保存しない。** 理由: (1) 全値が RaceEvaluation.features として評価結果側に記録され再現可能であるため、(2) 二重保存は「評価に使った値」と「保存された値」の乖離事故を生むため。ただし feature_schema_version は RaceEvaluation に必ず転記し、監査時は engine_version + feature_schema_version + config から再構築できることを再現性要件とする
- AIエンジンとの受け渡し方法: pipelines が `feature_builder.build(race, histories) -> FeatureSet` を呼び、`engine.evaluate(race, feature_set, config, now)` へ**引数として注入**する。グローバル変数・ファイル経由の受け渡しは禁止

## 3.4 RaceEvaluation（評価結果・システムの心臓）

**保存契約: 評価したら必ず1レース=1レコードで保存する。項目の欠損は空文字ではなくnullで表現し、「計算不能」と「未計算」を区別する。**

**Optional契約（v1.1.4）**: 「必(null許容)」の項目のOptional化は、レガシーデータとの互換性および未計算状態を表現するためである。Ver4以降の新規評価では、EvaluationEngineが必要な項目を設定する責務を持つ（新規評価での省略を許可するものではない。Ver4Engine実装時のテストで「新規評価でNoneでないこと」を検査する）。

**ParseError適用範囲（v1.1.4正式化）**: HitRecordCsvMapperにおいて、44互換列のうち「列自体が行に存在しない」（ヘッダー欠落・タイポ・破損）場合はParseError。「列は存在するがセルが空欄」はNoneとして受理する（レガシーデータの未記録表現）。LEGACY_DEFAULTS（拡張列不在時の補完定数）の適用対象は拡張メタデータ列のみとし、評価結果（スコア・指標）へ既定値を注入することを禁止する。

| 項目 | 型 | 必須 | 説明 |
|---|---|---|---|
| eval_id | str | 必 | `{race_date}_{venue_num:02d}_{race_number:02d}` |
| race_date / venue_num / venue_name / race_number / is_night | 上記 | 必 | Raceから複写 |
| engine_name | str | 必 | 例 "ver4"（②のエンジン識別。⑤5.2参照） |
| engine_version | str | 必 | 例 "4.2.0"（semver） |
| feature_schema_version | int | 必 | FeatureSetから転記 |
| model_version | str | 必(null許容) | MLモデル版 |
| evaluated_at | str | 必 | ISO8601 JST |
| danger_score | float | 必(null許容) | 現行 danger_score_v3 に相当。None=当時未記録 |
| danger_breakdown | json | 必(null許容) | 内訳 |
| upset_score | float | 必 | 再設計後スケール 0-100（現行0-9.5から移行） |
| upset_reasons | json | 必 | 荒れ理由リスト |
| rank_index | json | 必(null許容) | 6艇のランク指数 |
| featured_boats | json | 必(null許容) | 注目艇 |
| win_probs | json | 必(null許容) | ML勝率 6艇分 |
| race_type | str | 必 | 本命/中穴/大穴 分類 |
| hot_motor_score / awakening_score | float | 任 | 対象モーターがある場合 |
| local_advantage | json | 任 | 当地コース特徴量（Phase 3で接続） |
| features | json | 必 | 評価に使用したFeatureSetの全値（3.3と1:1） |
| match_index | float | 必(null許容) | upset_scoreから導出される評価結果（AI一致指数の近似値）。評価結果そのものであり、評価入力（weather等）とは区別する。**match_index is a derived evaluation metric. It is produced during evaluation and stored as part of RaceEvaluation. It is not recalculated by Mapper, Serializer, Repository, or presentation layers.** |

生成者: core/EvaluationEngine ／ 利用者: core/buyscore, output全部, ml ／ 保存先: evaluations/{date}.jsonl → Releases

## 3.5 Prediction（買い目予想）

| 項目 | 型 | 必須 |
|---|---|---|
| eval_id | str | 必 |
| pred_combo | str | 必 | 例 "1-2-3" |
| pred_prob / pred_ev / pred_odds | float | 必 |
| confidence | float | 必 |
| why_bet | str | 必 |
| patterns | json | 必 | 全候補（コンボ生成器出力） |

生成者: core/combo ／ 利用者: core/buyscore, output ／ 保存先: RaceEvaluationと同ファイルに内包

## 3.6 BuyDecision（購入判定）

| 項目 | 型 | 必須 |
|---|---|---|
| eval_id | str | 必 |
| purchased | bool | 必 |
| buyscore | float | 必(null許容) | None=当時未記録（BuyScoreエンジン導入前）。新規判定ではBuyEngineが必ず設定する |
| investment_type | str | 必 | 転がし/一発/通常/見送り |
| n_bets / cost | int | 必 |
| kelly_fraction | float | 必 |
| skip_reason | str | 任 | 見送り時必須 |
| config_version | str | 必 | buyscore_config の _version |

生成者: core/buyscore ／ 利用者: services(速報), storage(HitRecord), ml ／ 保存先: HitRecordへ合流

## 3.7 RaceResult（結果）

| 項目 | 型 | 必須 |
|---|---|---|
| eval_id | str | 必 |
| result_combo | str | 必 |
| payout | int | 必 |
| hit | bool | 必 |
| profit | int | 必 |

生成者: data/openapi_client（前日結果） ／ 利用者: ml/verification ／ 保存先: HitRecordへ合流

## 3.8 HitRecord（実績レコード＝現行hit_record.csvの後継）

RaceEvaluation＋Prediction＋BuyDecision＋RaceResult の結合ビュー。現行44列スキーマと1:1対応の互換列を維持する（⑯移行方法参照）。

v1.1.3追記: HitRecordは `weather: Optional[Weather]` を保持する。
**Weather is stored in HitRecord only for historical record reconstruction and legacy CSV compatibility. It is not the source of truth for evaluation. RaceEvaluation remains independent from Weather.**
責務の整理: Evaluation→Weatherを入力として受け取る（⑤5.2）／RaceEvaluation→Weatherを保持しない／HitRecord→当時の記録としてWeatherを保持する。

v1.1.3追記（列解釈の正式ルール）: HitRecordCsvMapperにおける「required columns」とは44互換列のみを指す。拡張列（engine_name等の末尾追加列）は前方互換の任意列であり、旧行に存在しないことは正常。44互換列の欠落はParseErrorとする。

生成者: storage/hit_record_store（各モデルの合流点） ／ 利用者: 検証・実績ページ・チューナー・学習エクスポート ／ 保存先: hit_record.csv（git管理外）＋Releases毎日退避

## 3.9 MotorHistory（1出走レコード）

現行 motor_history.csv の11列（date, venue_num, venue, motor_no, racer_no, racer_name, lane, place, ex_time, start_timing, race_number）をそのまま固定。生成者: features/motor_history ／ 保存先: motor_history.csv＋Releases。

## 3.10 VenueStatistics

venue_num, water_type, course_stats(json), venue_factor, updated_at。生成者: features/venue_stats ／ 利用者: features/feature_builder ／ 保存先: Releases。

## 3.11 LocalCourseStats（当地コース別成績）

現行 local_course_stats.csv の15列を固定（racer_no, venue_code, venue_name, course, starts, first..sixth, first_rate, top2_rate, top3_rate, last_updated）。

## 3.12 RankingEntry（日次ランキング1件）

| 項目 | 型 | 必須 |
|---|---|---|
| ranking_type | str | 必 | danger / hot_motor / awakening / manshuu |
| race_date / venue_name / race_number | - | 必 |
| score | float | 必 |
| rank_label | str | 必 | S/A/B等 |
| subject | json | 必 | 選手 or モーターの識別情報 |
| reasons | json | 必 |
| ai_comment | str | 必 |
| source_eval_id | str | 必 | **必ずRaceEvaluationを参照する（独自再計算禁止）** |

生成者: pipelines/ranking_daily（保存済み評価の抽出・整列のみ） ／ 保存先: rankings/{date}.json

## 3.13 NewsArticle（AI新聞）

race_date, sections(json: 注目レース/注目選手/警報/実績サマリー), generated_at, brand_version。生成者: output/html（保存済みデータのみから） ／ 保存先: note.html / note.md。

## 3.14 VerificationResult（検証結果）

race_date, n_races, n_purchased, n_hit, total_cost, total_payout, roi, by_race_type(json), by_rank(json), engine_version。生成者: ml/verification ／ 保存先: verification_history → Releases。

## 3.15 LearningData（学習用1行）

RaceEvaluation.features ＋ RaceResult を結合した特徴量行。生成者: ml/training_export ／ 保存先: training/{period}.csv → Releases。

## 3.16 SystemMetrics（システムKPI・新設）

ジョブ実行ごとに1レコード。AIの成績ではなく**システム自体の健康状態**を記録する。詳細は⑲。

| 項目 | 型 | 必須 | 説明 |
|---|---|---|---|
| metrics_id | str | 必 | `{date}_{job}_{run_id}` |
| race_date / job_name / run_id | - | 必 | Actionsのrun_idを記録 |
| started_at / finished_at / duration_seconds | - | 必 | 実行時間 |
| status | str | 必 | success / partial / failed |
| counters | json | 必 | ⑲19.2の計数項目 |
| rates | json | 必 | ⑲19.2の成功率項目 |
| errors | json | 必 | 例外件数・種別内訳（PlatformError 6分類別） |
| ai_summary | json | 任 | 当日ROI・的中率（verification由来の転記。再計算しない） |
| schema_version | int | 必 | |

生成者: pipelines（全ジョブが終了時に必ず出力） ／ 利用者: 監視・障害分析・長期統計・週次レポート ／ 保存先: system_metrics.json（当日）＋metrics/{month}.jsonl → Releases

## 3.17 モデル間関係図

```
Race ──┐
       ├─(features/feature_builder)──> FeatureSet
履歴DB ─┘                                  │
                                           ▼
                     EvaluationEngine.evaluate()
                                           │
Race ────────────> RaceEvaluation ──> Prediction ──> BuyDecision
      │                   │                                │
      │                   └──────────┬─────────────────────┘
      │                              ▼
RaceResult ─────────────────> HitRecord ──> VerificationResult
                                  │
                                  └──> LearningData ──> (再学習) ──> model.pkl
RaceEvaluation ──抽出──> RankingEntry ──> 画像/投稿文
HitRecord + RaceEvaluation ──> NewsArticle / 実績ページ
全ジョブ ──実行毎──> SystemMetrics ──> 監視・長期統計
```

---

# ④ 出力スキーマ固定

共通ルール（全出力に適用）:
- すべての保存JSONはトップレベルに `schema_version`（int）と `generated_at`（ISO8601 JST）を持つ
- **後方互換ルール: 項目の追加は自由、削除・改名・型変更は禁止**。削除・改名が必要な場合はschema_versionを+1し、旧項目を1バージョン間は併記する（deprecation期間）
- 読み手は未知キーを無視する（forward compatible）。書き手は必須キーを省略しない
- CSVは列追加のみ許可・末尾追加限定。列削除はマイグレーション（migration.py方式）必須

| 出力 | 保存形式 | 必須項目 | バージョン管理 |
|---|---|---|---|
| 評価データ | evaluations/{date}.jsonl（1行=1 RaceEvaluation） | ③3.4の必須列 | schema_version + engine_version + feature_schema_version |
| ランキング | rankings/{date}.json（4種を1ファイル） | ranking_type, entries[], source_eval_id | schema_version |
| 新聞 | note.html / note.md（成果物）＋sections JSON | ③3.13 | brand_version |
| 実績 | hit_record.csv（44列互換＋追加列）＋results.html | 44列全列 | .hit_record_schema_version（現行方式継続） |
| 学習データ | training/{period}.csv | features全列＋result | schema_version＋engine_version |
| システムKPI | system_metrics.json（当日）＋metrics/{YYYYMM}.jsonl | ③3.16の必須列 | schema_version |
| ログ | logs/{job}_{date}.log（テキスト、⑬形式） | ⑬参照 | なし（形式のみ固定） |
| キャッシュ | cache/{name}.json | schema_version, date, payload | dateが当日以外なら無効として破棄 |
| API出力（将来） | GET /evaluations/{date} 等、evaluations JSONLをそのまま返す | RaceEvaluation準拠 | URLに/v1/を含める |

互換性保証の要点: **表示層・将来APIは「evaluations JSONL」だけを読めば全成果物を再現できる**こと。これが崩れる変更は互換性破壊とみなし⑯の手続きを要する。

---

# ⑤ AIエンジン責務（core/）

## 5.1 担当範囲（これのみ）
1. 評価: EvaluationEngine.evaluate → RaceEvaluation
2. スコア: danger / upset / hot_motor / awakening / rank_index
3. 特徴量の消費: FeatureSet（③3.3）を受け取り評価に使う（特徴量の構築はL1）
4. 判定: race_type分類・注目艇選定
5. 買い目生成: patterns / combo（x_ai_rank_combo後継）
6. 購入判断: BuyScore・Kelly・転がし/一発・スキップ条件 → BuyDecision

## 5.2 EvaluationEngine 共通インターフェース（確定）

AIエンジンは以下の Protocol を実装する。**このシグネチャはFreeze対象であり、エンジン世代が変わっても変更しない。**

```python
class EvaluationEngine(Protocol):
    engine_name: str        # "ver4" / "ver5" / "ml_hybrid"
    engine_version: str     # semver "4.2.0"

    def evaluate(
        self,
        race: Race,
        feature_set: FeatureSet,
        weather: Optional[Weather],  # 評価入力。upset_score算出に使用（§3.1）。RaceEvaluationには保持しない
        config: dict,          # asahi_config相当（注入）
        now: datetime,         # 注入（内部取得禁止）
    ) -> RaceEvaluation: ...

    def predict(
        self,
        evaluation: RaceEvaluation,
        odds: OddsSnapshot | None,
        config: dict,
    ) -> Prediction: ...

class BuyEngine(Protocol):
    def decide(
        self,
        evaluation: RaceEvaluation,
        prediction: Prediction,
        config: dict,          # buyscore_config相当（注入）
        bankroll: BankrollState,
    ) -> BuyDecision: ...
```

固定事項:
- 実装系列: `Ver4Engine`（Phase 1で実装）→ `Ver5Engine` → `MLHybridEngine`。**入出力型（Race, FeatureSet → RaceEvaluation, Prediction, BuyDecision）はエンジンを差し替えても不変**
- ML予測器（win_probs）は `predictor: Callable` としてエンジンのコンストラクタに注入する。エンジン自身がモデルファイルをロードすることは禁止（⑨と整合）
- エンジンの選択は config/pipeline.json の `engine_name` キーで行う。pipelines がファクトリ `create_engine(engine_name)` で生成し、コード変更なしで差替え可能とする
- 新エンジンの本番投入条件: ⑭の並行稼働比較（旧エンジンとの出力突合＋バックテストROI比較）に合格すること。RaceEvaluation には engine_name / engine_version が常に記録されるため、世代混在期間もHitRecordで成績を分離集計できる
- **weatherは評価入力である**（v1.1初版の「評価には不使用」という記述は現行実装（upset_score算出でのwind_speed/wind_direction/wave_height参照）と乖離していたため、v1.1.1で訂正した）。RaceEvaluationはWeatherオブジェクトを保持しない。evaluate()の引数として都度渡し、その結果（upset_score・match_index等）のみをRaceEvaluationに記録する
- **match_indexはRaceEvaluationのフィールドとして保持する評価結果**（§3.4）。upset_scoreから導出される値であり、weather等の評価入力とは明確に区別する

## 5.3 禁止事項（違反はレビューで却下）
- ファイル読み書き（設定はdictで引数注入。asahi_config.jsonを開くのはpipelines）
- ネットワークアクセス
- 環境変数・Secretsの参照
- 現在時刻の取得（now は引数で受け取る）
- print以外の副作用（ロガーは注入されたものを使用）
- HTML/画像/メール/X/GitHub/CSVへの一切の関与
- FeatureSetを経由しない生データ（RaceEntryの独自加工）による判定

## 5.4 純粋性の定義
同一の（Race, FeatureSet, config, now）に対して常に同一の RaceEvaluation / BuyDecision を返す。乱数を使う場合はseedを引数で受け取る。この性質が⑭の回帰テスト・並行稼働比較・エンジン世代間比較を可能にする。

---

# ⑥ 保存層責務（storage/）

| 対象 | 保存タイミング | 読込タイミング | 更新方法 | 整合性保証 | バックアップ | 復旧方法 |
|---|---|---|---|---|---|---|
| evaluations JSONL | 評価直後（レース単位で追記） | 表示・ランキング・検証時 | 追記のみ（上書き禁止） | eval_id一意性チェック | 日次でReleasesへ | Releasesから該当日を取得 |
| hit_record.csv | BuyDecision確定時に追記、結果判明時に該当行更新 | 検証・実績・チューナー | 追記＋eval_idキーで結果列のみ更新 | schema_version検証＋data_integrity検証を書込前後に実行 | **書込成功＝Releases退避＝git commitを1トランザクション扱い（いずれか失敗で全体失敗としERROR通知）** | Releases最新版→git履歴の順で復旧 |
| motor_history.csv | 朝の履歴更新ジョブ | ランキング・学習 | 追記 | (date,venue,race,lane)一意 | 日次Releases | 同上 |
| local_course_stats.csv | 週次再構築 | 特徴量生成 | 全置換（進捗ファイルで中断再開） | k_race_history_integrity | 構築毎にReleases | 進捗ファイルから再開 |
| 設定JSON | 手動＋チューナー | 各pipeline起動時に1回 | チューナーは_versionを+1して書込 | JSONスキーマ検証 | git管理（変更履歴=git log） | git revert |
| モデル(pkl) | 週次再学習後 | pipeline起動時 | Releasesへ新タグ、latestポインタ更新 | 学習時メトリクス閾値を満たさなければ反映しない | Releases世代保管（削除しない） | latestを前世代へ戻す |
| system_metrics | 全ジョブ終了時（必須・失敗時も出力） | 監視・週次レポート・障害分析 | 当日jsonは全置換、月次jsonlは追記 | metrics_id一意 | 月次でReleasesへ | 欠損許容（計測データのため）。ただし欠損自体をWARNING |
| キャッシュ | 各ジョブ | 同日後続ジョブ | 全置換 | date検証（当日以外破棄） | 不要（再生成可能） | 再生成 |
| 冪等性記録 | 配信直前 | 配信直前 | 追記 | (channel, message_key)一意 | 日次Releases | Releases復元。復元不能時は「当日分は送信済み扱い」に倒す（二重送信より欠送を選ぶ） |
| ログ | 常時 | 障害調査時 | 追記 | なし | Actionsアーティファクト7日 | なし |

原則: **「GitHub Actionsコンテナはステートレス」を前提に、ジョブ終了時に消えて困るものは必ずこの表のバックアップ列に従って退避する。** 表にない新ファイルの作成はレビュー必須。

---

# ⑦ 表示層責務（output/）

- 入力は storage の保存済みデータ（evaluations / hit_record / rankings / verification / system_metrics）**のみ**
- AI判定・スコア計算・API呼出を一切行わない。四則演算は「表示のための集計」（合計・平均・件数・ROI表示値）に限定する
- 生成対象と入力の対応（固定）:

| 成果物 | 入力 | 出力ファイル |
|---|---|---|
| AI新聞 | evaluations + rankings + verification(前日) | note.html / note.md |
| ランキング画像 | rankings/{date}.json | danger.png, hot_motor.png, awakening.png, manshuu.png |
| 実績ページ（公開） | hit_record + verification | ai_result_public.html |
| 実績ページ（開発） | hit_record + verification + buyscore_log + system_metrics | ai_result_developer.html |
| メール本文 | 上記いずれかの成果物データ | テキスト |
| X投稿文 | rankings / verification | テキスト |

- 全成果物はテンプレート＋brand設定（⑮）から生成する。色・アイコン・閾値表示のハードコード禁止
- 同一入力から何度生成しても同一出力（配信可否は表示層の関知外）

---

# ⑧ 配信層責務（services/）

- 担当: メール送信（Gmail）／GitHub Pages反映（成果物commit&push）／（将来）X投稿／障害通知
- 禁止: 評価・計算・本文組み立て（本文はL4から完成品で受け取る）
- **冪等性はここで一元管理する**。送信前に `idempotency.check(channel, message_key)` を必ず通す
  - message_key 規約: `{channel}:{job}:{race_date}:{対象ID}`（例 `mail:arashi:20260713:12_05`）
  - 現行の sent_*.txt / manshuu_sent.json / korogashi_cache.json の3方式はこの1方式に統合する
- 送信失敗時: ⑫のリトライ規定に従う。リトライ枯渇時は冪等性記録に**書き込まず**終了（次回再送のため）
- 配信結果（成功/失敗件数）はSystemMetricsのcountersへ計上する
- 配信時間帯・宛先・チャネル有効/無効はすべて⑮の設定ファイルで制御する

---

# ⑨ 学習パイプライン（ml/）

```
[毎晩] verify_results:
  RaceResult取得(L0) → HitRecord更新(L3) → VerificationResult生成・保存
       → buyscore_tuner（config _version+1で提案。自動反映は閾値内変更のみ、
          閾値超は提案ログのみ出し手動承認。※Feature Freeze期間中は⑳に従い停止）

[週次] retrain_weekly:
  1. training_export: HitRecord + evaluations → LearningData CSV
  2. 学習: scikit-learn 1.5.2 / LightGBM 4.5.0（requirements-ml.txt固定を維持）
  3. 評価: ホールドアウトでAUC・的中率・ROIシミュレーション
  4. ゲート: 現行モデルの評価値を下回る場合は保存のみ・本番反映しない
  5. 保存: Releasesへ model_{version}.pkl、合格時のみ latest 更新
  6. 記録: 学習メトリクス（学習時間・AUC・データ件数）を improvement_log と SystemMetrics へ
```

AI本体との接続: core はモデルを直接ロードしない。pipelines が起動時に storage 経由で latest モデルを取得し、predictor として EvaluationEngine のコンストラクタに**注入**する（⑤5.2）。これによりモデル差替えが core 無変更で可能になり、⑭の「モデル固定での回帰テスト」も成立する。

---

# ⑩ ディレクトリ構成固定（最終形）

```
my-app/
├── models/        # データモデル定義（dataclass）。全レイヤーが参照可
├── core/          # L2 AIエンジン
│   ├── engine.py  #   EvaluationEngine Protocol / create_engineファクトリ
│   ├── ver4/      #   Ver4Engine（scoring: evaluation.py, danger.py, upset.py,
│   │              #   hot_motor.py, awakening.py）
│   ├── buyscore/  #   buyscore.py, kelly.py, korogashi.py（BuyEngine実装）
│   └── combo.py
├── data/          # L0 外部取得
│   ├── openapi_client.py, beforeinfo.py, odds.py, fan_file.py
│   └── kfile/     #   downloader.py, lzh.py, race_parser.py, payout_parser.py
├── features/      # L1 特徴量
│   ├── feature_builder.py   # FeatureSetの唯一の生成地点
│   ├── motor_history.py, venue_stats.py
│   └── local_course/
├── storage/       # L3 保存
│   ├── exceptions.py（PlatformError/ParseError/StorageError、v1.1.7反映）
│   ├── repositories/  # evaluation_repository.py, hit_record_repository.py
│   │              #   （旧記載evaluation_store.py/hit_record_store.pyを実態へ是正、v1.1.7）
│   ├── durability.py  # 書込=Releases退避=commit不可分化（v1.1.7追加）
│   ├── clients/   #   protocols.py, github_release_client.py, git_client.py（v1.1.7追加）
│   ├── release_storage.py, cache.py, idempotency.py, metrics_store.py
│   ├── mappers/   #   モデル⇔行(dict)の純粋変換。row_types.py, hit_record_mapper.py,
│   │              #   motor_history_mapper.py, local_course_stats_mapper.py,
│   │              #   learning_data_mapper.py（Step2で追加、v1.1.2）
│   ├── serializers/ # モデル⇔JSON互換dictの純粋変換（Step2で追加、v1.1.2）
│   └── schema/    #   schema_version.py, migration.py, integrity.py
├── output/        # L4 表示
│   ├── html/      #   news.py, results_public.py, results_developer.py, templates/
│   ├── image.py
│   └── text/      #   mail_bodies.py, post_texts.py
├── services/      # L5 配信
│   ├── mailer.py, publisher.py, notifier.py
├── pipelines/     # ymlから呼ぶ唯一のエントリーポイント群
│   ├── arashi_watch.py, morning_korogashi.py, manshuu_watch.py
│   ├── ranking_daily.py, verify_results.py, retrain_weekly.py
├── ml/            # L6 学習
│   ├── training_export.py, retrain.py, verification.py, tuner.py
├── config/        # asahi_config.json, buyscore_config.json, brand.json,
│                  # delivery.json, pipeline.json
├── analysis/      # backtest_roi.py, backtest_optimizer.py, optimize_threshold.py
├── tools/         # 保守スクリプト（check系で残すもの）
├── tests/         # unit/ integration/ regression/
├── docs/          # 本設計書, scoring_spec.md, incidents/
├── logs/          # git管理外
└── .github/workflows/
```

## 依存ルール（確定・CIで検査する）

| ディレクトリ | import してよいもの | 禁止 |
|---|---|---|
| models | 標準ライブラリのみ | 全レイヤー |
| core | models, config(値のみ) | data, storage, output, services, requests, ファイルI/O |
| data | models | core, output, services |
| features | models, data, storage | core, output, services |
| storage | models | core, output, services |
| output | models, storage(読取API), config | core, data, services |
| services | models, storage(idempotency, metrics), config | core, data, output(生成関数) |
| pipelines | 全部（唯一の結線地点） | - |
| ml | models, storage, core(評価再現用) | output, services |
| tools/analysis | 全部（ただし本番pipelineからimport禁止） | - |

## 配置ルール
- ymlが直接呼べるのは pipelines/ 配下のみ
- ルート直下への .py 新規配置は禁止（現行の平置き構造の再発防止）
- 一時スクリプトは tools/ のみ。2週間参照されないものは削除する

## Mapper層ルール（v1.1.2追加）

**HitRecordCsvMapper Design Rules**
- This mapper is the only mapper allowed to aggregate multiple domain models.
- All other mappers remain one-to-one mappings.
- No business logic is allowed.
- Missing required columns raise ParseError.
- Unknown columns must not be silently ignored unless explicitly designated as forward-compatible extension columns.

補足:
- 「複数ドメインモデルの集約」が許されるのはHitRecordCsvMapper（HitRecord＝RaceEvaluation/Prediction/BuyDecision/RaceResultの集約モデル、③3.8）のみ。他の全Mapperは1モデル⇔1行の変換に限定する
- 未知列の黙殺禁止は、タイポ・列名変更・CSV破損の早期検知が目的。前方互換の拡張列として明示的に指定された列集合（Mapperの定数として宣言）のみ、未知でも許容できる
- 必須列の欠落・JSON列の破損はParseError（⑫の例外6分類）を送出する。サイレント失敗の全面禁止（⑫）と整合

---

# ⑪ 命名規則（確定）

| 対象 | 規則 | 例 |
|---|---|---|
| Pythonファイル/モジュール | snake_case、`x_` 等の接頭辞禁止、省略禁止 | evaluation_store.py（× eval_st.py） |
| 関数 | snake_case、動詞始まり | build_race_evaluation, calc_danger_score |
| 変数 | snake_case、意味の分かる完全語 | race_number（× rno, r） |
| クラス/dataclass | PascalCase、単数形 | RaceEvaluation, Ver4Engine |
| 定数 | UPPER_SNAKE_CASE | REQUEST_TIMEOUT_SECONDS |
| Enum | クラスPascalCase・メンバーUPPER | RaceType.HONMEI |
| CSV | snake_case英語、日本語ヘッダー禁止（表示層で和訳） | hit_record.csv, motor_history.csv |
| CSV列名 | snake_case、単位を名前に含める | wind_speed_mps, avg_st |
| JSON | ファイルsnake_case、キーsnake_case | evaluations/20260713.jsonl, system_metrics.json |
| HTML | snake_case | ai_result_public.html |
| 画像 | `{ranking_type}.png` | danger.png, hot_motor.png |
| 設定ファイル | `{領域}_config.json` ＋ brand.json / delivery.json / pipeline.json | buyscore_config.json |
| ログ | `{job}_{YYYYMMDD}.log` | arashi_watch_20260713.log |
| ディレクトリ | 単語1つの複数形または役割名 | pipelines, storage |
| Releasesアセット | `{name}_{version|date}.{ext}` | model_v4.2.0.pkl, hit_record_20260713.csv |
| エンジン | `Ver{N}Engine` / `MLHybridEngine`、engine_name は小文字 | Ver4Engine / "ver4" |
| git ブランチ | `phase{N}/{短い説明}` | phase1/core-engine |
| コミット | Conventional Commits（feat:/fix:/refactor:/docs:） | 現行慣行を継続 |

既存名の移行対応表（代表）: notify_arashi→pipelines/arashi_watch、x_asahi_scoring→core/ver4、x_buyscore→core/buyscore、x_note_report→output/html/news、x_release_storage→storage/release_storage、x_verification→ml/verification。

---

# ⑫ エラー設計

## 例外方針
- 例外クラスは `PlatformError` を基底に `DataFetchError / ParseError / StorageError / DeliveryError / ConfigError / ModelError` の6種のみ
- core は例外を投げてよいが握りつぶし禁止。捕捉と方針決定は pipelines のみが行う
- **サイレント失敗の全面禁止**: `except: pass` 禁止。無視する場合も必ずWARNINGログ（motor.csv 0行事故の再発防止）
- 全例外は種別ごとに件数をSystemMetrics.errorsへ計上する

## リトライ・タイムアウト（確定値）

| 対象 | タイムアウト | リトライ | 間隔 |
|---|---|---|---|
| OpenAPI / beforeinfo | 15秒 | 3回 | 指数 2/4/8秒 |
| GitHub API (Releases) | 30秒 | 3回 | 指数 |
| Gmail送信 | 30秒 | 2回 | 30秒 |
| KファイルDL | 60秒 | 2回 | 60秒 |

## ジョブレベルのフェイルセーフ
- 1レースの評価失敗は記録してスキップし、ジョブ全体は続行（現行 `|| echo 続行` 方針を構造化）
- ただし「全レース失敗」「hit_record書込失敗」「モデルロード失敗」はジョブをERROR終了させ通知する
- 部分成功の定義: 成功率80%以上=WARNING継続、未満=ERROR。判定結果はSystemMetrics.statusへ記録（success/partial/failed）

## 通知条件
- ERROR発生 → 運用者へメール通知（1ジョブ1通に集約）
- 同一ERRORが3日連続 → 件名に[連続障害]を付与
- 「レース0件」はWARNING扱いだが、開催日で0件が2窓連続したらERROR（fetch_programs事故の再発検知）

## 復旧条件
- 復旧手順は⑥の表の「復旧方法」列に従う。手順書外の手作業復旧を行った場合は docs/incidents/ に記録を残す

---

# ⑬ ログ設計

形式（全ジョブ統一・1行1イベント）:
```
2026-07-13T07:30:15+09:00 [INFO] [ranking_daily] [20260713_12_05] danger_score=78.5 rank=S
{ISO8601 JST} [{LEVEL}] [{job}] [{eval_id または -}] {メッセージ key=value...}
```

| レベル | 基準 | 例 |
|---|---|---|
| DEBUG | 開発・調査時のみ。中間値・分岐理由 | 各スコアの内訳、スキップ判定の根拠 |
| INFO | 正常系の節目。1レース1行まで | 評価完了、保存完了、送信完了、ジョブ開始/終了(件数付き) |
| WARNING | 継続可能な異常。**黙認する異常は必ずここに出す** | 1レース取得失敗、フォールバック使用、0行データ検出 |
| ERROR | ジョブ失敗・データ欠損リスク・通知対象 | hit_record書込失敗、全件取得失敗、モデルロード失敗 |

- ジョブ終了時に必ずサマリー行を出す: `[INFO] [job] [-] summary races=45 evaluated=44 purchased=6 errors=1`（この値はSystemMetricsと一致させる）
- 秘匿情報（APIキー・メールアドレス・トークン）のログ出力禁止（⑰）
- ログはActionsアーティファクトとして7日保持

---

# ⑭ テスト戦略（最重要）

## テスト種別

| 種別 | 対象 | 場所 | 実行タイミング |
|---|---|---|---|
| 単体テスト | core全関数（純粋関数なので入出力固定で検証） | tests/unit | 毎push（CI必須化） |
| 結合テスト | pipelines を stub data + tmp storage で通し実行 | tests/integration | 毎push |
| 回帰テスト | ゴールデンデータ比較（下記） | tests/regression | 毎push＋Phase切替前 |
| 並行稼働比較 | 新旧システム同時実行・出力突合 | 本番Actions | Phase 2/4切替前 各1週間以上 |

## ゴールデンデータ回帰テスト
- 過去実データから **100レース（初期セット）→1000レース（Phase 2完了まで拡充）** を固定入力として保存
- 旧システム（現行コード＝⑳でFreezeされた状態）の出力を「正」としてスナップショット化し、新coreの出力と比較する

## 一致基準（Phase切替の合格ライン・確定値）

| 比較項目 | 基準 |
|---|---|
| danger_score 一致 | 100%（小数第2位まで一致） |
| upset_score 一致 | スケール移行前100%／移行後は順位相関0.99以上＋新旧対応表の全件突合 |
| rank_index・featured_boats 一致 | 100% |
| BuyScore・purchased判定 一致 | 100% |
| pred_combo・patterns 一致 | 100% |
| hit_record CSV 一致 | 44互換列で100%（浮動小数は1e-6許容） |
| ランキング（4種の順位と件数） | 100% |
| 新聞 | 数値・選手名・レース選定100%（文面テンプレ差は許容し、差分はレビュー承認） |
| 検証集計（ROI・的中率） | 100% |

## Phase切替条件（共通）
1. 回帰テスト全件グリーン
2. 並行稼働7日間で上記一致基準を満たす（不一致は原因を特定し「新が正しい」場合のみ基準表を更新して承認記録を残す）
3. ⑥のバックアップ・復旧手順を1回実演（リハーサル）済み
4. ロールバック手順（ymlの呼び先を旧pipelineへ戻す）を文書化済み
5. 切替対象ジョブのSystemMetricsが並行稼働期間中 status=success 率95%以上

---

# ⑮ 設定ファイル設計

原則: **「数字・文言・時刻・色は設定へ、アルゴリズムはコードへ」**。以下はコード変更なしで変更可能でなければならない。

| ファイル | 内容 | 変更者 |
|---|---|---|
| config/asahi_config.json | 評価エンジン全閾値・重み（現行構造を維持: danger_score, upset_prob, boat_relative_score, lane_rank_scores, featured_racers） | 手動 |
| config/buyscore_config.json | weights, thresholds, odds_band_bonus, race_type_points, star_thresholds, korogashi, ippatsu, kelly, capital_management, tuning（現行構造維持） | 手動＋チューナー |
| config/brand.json | システム名, AI_VERSION表示, ランク閾値と色, アイコン(⏱⚡等), 表示順, 文言テンプレート | 手動 |
| config/delivery.json | チャネル有効/無効, 宛先, 配信時間帯, 1日投稿数上限, 公開/非公開設定, DRY_RUN | 手動 |
| config/pipeline.json | **engine_name（エンジン差替え）**, ランキング件数, 監視間隔パラメータ, 対象レース絞込(ranking_filter統合), 締切前分数, メトリクス閾値 | 手動 |

- 全設定はトップレベルに `_version`（int）と `_comment` を持つ
- 起動時にスキーマ検証し、不正ならConfigErrorで即終了（黙って既定値にしない）
- Secrets（⑰）は設定ファイルに書かない

---

# ⑯ バージョン管理方針

| 対象 | 方式 | 互換性 | 移行 | ロールバック |
|---|---|---|---|---|
| 設計書（本書） | docs/でgit管理。変更はPR＋変更履歴表を末尾に追記 | - | - | git revert |
| 設定JSON | `_version` 単調増加。チューナー変更もgit commit | 未知キー無視 | 起動時に旧版検出→既定値補完＋WARNING | git revert |
| CSVスキーマ | .hit_record_schema_version（現行方式）を全CSVへ拡張: `.{name}_schema_version` | 列追加のみ後方互換 | migration.py方式（自動バックアップ→変換→検証） | バックアップCSVへ復元 |
| 評価JSONL | schema_version＋engine_version＋feature_schema_version を各行に記録 | 追加自由・削除は+1 | 読み手が版分岐 | 旧版読取コードを1版残す |
| FeatureSet | feature_schema_version 単調増加。キー追加=+1、キー削除は禁止（nullで残す） | 未知キー無視 | 学習データ再エクスポートで追随 | 旧版キー集合で再構築 |
| MLモデル | Releasesに model_{semver}.pkl、latestタグ | ライブラリ版固定(sklearn1.5.2/LGBM4.5.0)が前提 | ゲート合格時のみlatest更新 | latestを前世代へ付替え |
| AIエンジン | engine_name＋engine_version（semver）。スコア計算式の変更=minor+1、互換性破壊=major+1、**新世代（Ver5等）=新engine_name** | RaceEvaluationに常に記録。EvaluationEngineインターフェース（⑤5.2）は不変 | 並行稼働比較(⑭)必須 | config/pipeline.json の engine_name を旧値へ戻すのみ |

**再現性の保証**: 任意のHitRecord行は engine_name + engine_version + model_version + feature_schema_version + config _version から当時の判定を再現できること。これを監査可能性の要件とする。

---

# ⑰ セキュリティ設計

| 対象 | 方針 |
|---|---|
| APIキー・トークン | GitHub Actions Secrets のみ。コード・設定JSON・ログへの記載禁止。bitget_config.py型の平文キーは全廃 |
| Secrets の受け渡し | 環境変数→pipelinesが読み、必要なserviceにのみ注入。coreは参照禁止(⑤) |
| GitHub | リポジトリはprivate維持。公開が必要な成果物（実績ページ等）のみPagesへ。PATは最小権限（contents:write のみ） |
| 個人情報 | 配信先メールアドレスはSecrets管理。ログ・コミットへの出力禁止。選手情報は公開データのみ扱う |
| メール | 送信はGmailアプリパスワード（Secrets）。本文に内部エラー詳細・スタックトレースを含めない（運用者向け通知のみ可） |
| 権限分離 | 無関係コード（仮想通貨ボット）と取引所APIキーは本リポジトリから排除（Phase 5で分離） |
| ログ・メトリクス | ⑬の秘匿ルール。system_metricsにも個人情報・キー情報を含めない。アーティファクト保持7日で自動失効 |
| 依存パッケージ | requirements固定を維持。追加時はレビュー必須 |

---

# ⑱ Phase 1 開始条件

以下**すべて**が満たされた時点でPhase 1（コアエンジン作成）を開始する。1つでも未確定ならPhase 1に着手しない。

- [ ] 本設計書（1.1版）がリポジトリ docs/ にコミットされ、確定版として承認されている
- [ ] データモデル（③）の全項目・型・必須が確定し、追加の未決事項がない（FeatureSet・SystemMetricsを含む）
- [ ] EvaluationEngine / BuyEngine インターフェース（⑤5.2）が確定している
- [ ] レイヤー責務（⑤〜⑧）と依存ルール（⑩）が確定している
- [ ] 命名規則（⑪）と既存名の移行対応表が確定している
- [ ] ディレクトリ構成（⑩）が確定している
- [ ] ログ形式（⑬）とエラー方針（⑫）が確定している
- [ ] テスト戦略（⑭）の一致基準・Phase切替条件が数値で確定している
- [ ] 設定ファイル5種（⑮）の構成が確定している
- [ ] 移行方法・バックアップ・ロールバック（⑥⑯）が文書化されている
- [ ] **Feature Freeze（⑳）が宣言され、凍結タグが打たれている**
- [ ] ゴールデンデータ100レース分の入力データが確保されている（過去実データから抽出可能なことを確認済み）
- [ ] hit_record.csv 欠落期間（2026-07-04以降）の復旧可否が確定している（復旧不能なら「欠落期間」として記録し確定させる）
- [ ] 現行システムの全資産（Phase 0設計書⑨）がReleasesまたはgitで保全済みである

---

# ⑲ システムKPI（運用メトリクス）設計

## 19.1 目的

AIの成績（ROI・的中率）とは別に、**システム自体の健康状態**を毎ジョブ記録する。用途は (1) 障害の早期検知（監視）、(2) 性能改善（実行時間・成功率の推移）、(3) 障害分析（いつから壊れたかの特定）、(4) 長期統計。motor.csv 0行・hit_record欠落のような「静かな劣化」を数値で可視化することが第一目的である。

## 19.2 記録項目（確定・SystemMetrics ③3.16に格納）

**counters（計数）**
- races_found / races_evaluated / races_skipped（評価レース数）
- purchase_candidates / purchases_decided（購入対象数）
- rankings_generated（4種別内訳付き）
- records_written（hit_record / evaluations / motor_history 別の保存件数）
- deliveries_attempted / deliveries_sent（チャネル別）
- news_generated（新聞生成 0/1）

**rates（成功率・0.0-1.0）**
- api_fetch_success_rate（OpenAPI/beforeinfo/オッズ別）
- storage_success_rate（GitHub Releases保存成功率を含む）
- delivery_success_rate

**timings（秒）**
- job_duration_seconds（Actions実行時間）
- news_generation_seconds（新聞生成時間）
- training_seconds（学習ジョブのみ）
- api_fetch_seconds_p50 / p max

**errors**
- 例外件数（PlatformError 6分類別）＋最後のエラーメッセージ要約

**ai_summary（転記のみ・再計算禁止）**
- 当日ROI・的中率・購入件数（VerificationResultからの複写）

## 19.3 保存・利用ルール

- 全pipelinesジョブは**成功・失敗を問わず**終了時にSystemMetricsを1件出力する（finallyで保証）。出力できない事態はログのERRORのみ許容
- 当日値: system_metrics.json（当日全ジョブの配列・全置換）／長期: metrics/{YYYYMM}.jsonl 追記→月次Releases退避
- 閾値監視: config/pipeline.json の metrics閾値（例 api_fetch_success_rate < 0.8 でWARNING、< 0.5 でERROR通知）に基づき notifier が判定する。判定はservices層の責務であり、core・outputはメトリクスを解釈しない
- 週次レポートと開発用実績ページ（⑦）はこのデータを表示する

---

# ⑳ Feature Freeze（現行システム凍結宣言）

## 20.1 宣言

**2026-07-13 をもって、現行システム（notify_arashi系一式）を Feature Freeze とする。**

現行システムはPhase 1〜4の間、新システムの**比較基準（リファレンス実装）**として扱う。比較基準が動くと⑭の回帰テスト・並行稼働比較が成立しないため、以下を禁止する。

## 20.2 禁止事項（凍結期間中）

1. **新機能追加の禁止**（新しいスコア・新しい配信・新しいページ・新しい列の追加を含む）
2. **アルゴリズム変更の禁止**（評価式・判定式・買い目生成・BuyScore計算の変更）
3. **スコア変更の禁止**（閾値・重み・係数の変更。asahi_config.json / buyscore_config.json の変更を含む）
4. buyscore_tuner による設定自動反映の**停止**（提案ログ出力のみ継続する。tuningの再開は新システム切替後）
5. スキーマ変更の禁止（hit_record.csv への列追加・削除）
6. リファクタリング・ファイル名変更・ファイル移動の禁止

## 20.3 例外的に許可される変更

| 許可対象 | 条件 |
|---|---|
| 稼働継続に必須の障害修正（API仕様変更対応・取得失敗の修正など） | 修正前に docs/incidents/ へ記録し、修正がスコア出力に影響する場合はゴールデンデータのスナップショットを再生成して差分を承認記録に残す |
| セキュリティ修正 | 即時適用可。事後記録必須 |
| ログ追加（出力値を変えない観測のみ） | WARNING/デバッグログの追加のみ可 |
| データ復旧作業（hit_record欠落期間対応など） | ⑥の復旧手順に従う |

上記以外の変更要求はすべて「新システム側（Phase 1以降）への要件」として docs/ に積み、現行には適用しない。既知課題（upset_scoreスケール・地元選手機能・x_ai_rank_combo接続）も同様に**新システム側でのみ**実施する。

## 20.4 凍結の実施手続き

1. 現行 main ブランチに git タグ `freeze-v1-baseline` を打つ
2. このタグ時点のコードでゴールデンデータ（⑭）のスナップショットを生成する
3. 凍結時点の設定ファイル（asahi_config.json / buyscore_config.json）のハッシュを docs/ に記録する
4. 凍結解除は「Phase 5完了（旧コード削除）」をもって自動的に成立する

## 20.5 凍結期間

開始: 2026-07-13（本書承認日） ／ 終了: Phase 5 完了時

---

# 付録: 本書の変更履歴

| 版 | 日付 | 変更内容 |
|---|---|---|
| 1.0 | 2026-07-13 | 初版制定（Architecture Freeze） |
| 1.1 | 2026-07-13 | 最終版: ③にFeatureSet(3.3)・SystemMetrics(3.16)を追加、⑤にEvaluationEngine/BuyEngineインターフェース(5.2)を追加、⑲システムKPI・⑳Feature Freezeを新設、④⑥⑦⑧⑨⑫⑬⑭⑮⑯⑰⑱へ関連反映 |
| 1.1.1 | 2026-07-13 | Step2着手時の実コード調査（notify_arashi.py）により判明した設計書と現行実装の乖離を修正。③3.1 Race.weatherの記述を「評価に不使用」から「upset_score算出の評価入力」へ訂正。③3.4 RaceEvaluationにmatch_index（float・必須）を追加。⑤5.2 EvaluationEngine.evaluate()のシグネチャにweather引数を追加（RaceEvaluationはWeatherを保持しない設計は維持）。契機: Step2 Mapper設計におけるhit_record.csv列（wind_speed/wind_dir/wave/match_index）の対応先調査。match_indexの生成責務ルール（evaluation時にのみ生成、Mapper/Serializer/Repository/表示層での再計算禁止）を③3.4へ明記 |
| 1.1.2 | 2026-07-13 | ⑩へstorage/mappers/・storage/serializers/を追記（Step2計画の論点④で承認済み、v1.1.1で反映漏れだったものを本版で反映）。⑩へ「Mapper層ルール（HitRecordCsvMapper Design Rules）」を新設: 複数モデル集約はHitRecordCsvMapperのみ許可・他Mapperは1:1限定・ビジネスロジック禁止・必須列欠落はParseError・未知列の黙殺禁止（前方互換拡張列として明示指定された列を除く） |
| 1.1.3 | 2026-07-13 | ③3.8 HitRecordへ weather: Optional[Weather] を追加（案W-B）。目的を「歴史的記録の再構築とレガシーCSV互換」に限定する英文ルールを明記。RaceEvaluationのWeather非保持は維持。「required columns=44互換列のみ、拡張列は任意」の列解釈を正式ルール化。契機: Step2-3でwind_speed/wind_dir/wave列の保持先欠如を検出 |
| 1.1.4 | 2026-07-13 | レガシー実データ検証（danger_score_v3全255行空欄等）に基づく是正。③3.4の7項目（danger_score, danger_breakdown, rank_index, featured_boats, win_probs, match_index, model_version）を「必(null許容)」へ変更しOptional契約を明文化（新規評価での設定責務はEvaluationEngine）。ParseError適用範囲（列欠落=ParseError／空欄セル=None受理）とLEGACY_DEFAULTSの適用制限（拡張メタデータのみ・評価結果への注入禁止）を正式化 |
| 1.1.5 | 2026-07-13 | ③3.6 BuyDecision.buyscoreを「必(null許容)」へ変更（実データ165/255行が空欄）。コレクション型ポリシーを正式化: Optionalは「値が存在しない」の表現に限定し、空集合で自然に表現できるコレクション型（upset_reasons, patterns等のtuple）には適用せず空タプル()で表現する |
| 1.1.6 | 2026-07-13 | ③3.6 BuyDecision.kelly_fractionを「必(null許容)」へ変更（LEGACY_DEFAULT 0.0注入は「未計算」と「計算結果0.0」を区別できないため不採用）。レガシー=None、新規判定はBuyEngineが必ず設定する契約。Step2-3レビューShould項目のうちfeature_schema_version一致前提・patternsのJSON互換前提をdocstringへ明文化 |
| 1.1.7 | 2026-07-14 | Step3計画承認に伴う⑩更新: durability.py・clients/を追加。Step2-6実装との乖離（evaluation_store.py/hit_record_store.py記載 vs 実装repositories/）を実態へ是正し記録。Releasesアセット命名規約（最新=`{name}.{ext}`／スナップショット=`{name}_{YYYYMMDD}.{ext}`、新タグdata-store-v2、現行data-storeタグへの書込禁止）を確定 |
| 1.1.8 | 2026-07-14 | アセット命名の日付基準を明記: スナップショットの{YYYYMMDD}は**JST基準**（データ内容の開催日race_date由来。システム実行時刻ではない）。GithubReleaseClientのリトライ方針をRetryPolicyとして本体から分離（⑫の確定値: タイムアウト30秒・3回・指数バックオフを既定値とする） |
