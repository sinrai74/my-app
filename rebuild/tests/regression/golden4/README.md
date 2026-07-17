# Ver4 Golden Input/Output 生成手順（Step4-0）

## 進捗区分（Step4-0レビュー承認済み）

- **Step4-0A: Wrapper・回帰基盤 — 完了**（tools/golden_wrapper.py、test_ver4_golden.py、本README）
- **Step4-0B: Windows運用環境でのGolden Input/Output生成・SHA-256固定 — 未完了**
  （本手順に従い運用者環境で実施。完了後にStep4-0B完了報告を提出する）

## 厳守ルール（Step4-0Bレビュー指示）

1. **Golden生成はFreezeタグで実施すること**: 生成実行時のリポジトリは
   `freeze-v1-baseline` タグの状態（またはその時点と同一のscoringコード）であること。
   `--source-commit` にはそのコミットIDを渡す（省略時はHEADを自動記録するため、
   HEADがfreezeタグ位置にあることを確認してから実行する）。
2. **生成後はSHA-256を固定し、以後は承認記録なしに更新しないこと**:
   manifest.jsonに記録されたハッシュが基準値となる。回帰テストが照合し、
   不一致はFailする。正当な更新（意図的な再ベースライン）は、承認記録
   （設計書⑳20.3）とともに再生成した場合のみ許される。

このディレクトリは、Ver4評価エンジン移植（Step4）の回帰基準となる
**Golden Input（入力100件）と Golden Output（期待出力100件）** を格納する。

## 目的

移植先の `Ver4Engine` の出力が、現行 `build_race_evaluation_v4` の出力と
**完全一致（数値許容1e-6）** することを保証する。Step2のGolden CSV（出力側）に対し、
本Goldenは **入力→出力** の一致を見るためのもの。

## 重要な制約

- 現行 `x_asahi_scoring.build_race_evaluation_v4` は **一切変更しない**。
  抽出は `tools/golden_wrapper.py` が旧関数を **外側から import して呼ぶ** ことで行う。
- 生成された Golden 一式（inputs/ expected/ manifest.json）は **変更禁止**。
  回帰テストが manifest 記録の SHA-256 と実ファイルを照合し、改変を検出する（C7）。
- 正当な更新（例: 意図的な仕様変更後の再ベースライン）を行う場合のみ、
  承認記録とともに再生成し、manifest のハッシュを更新する（設計書⑳20.3）。

## 生成手順

Golden の生成には **freeze-v1-baseline 時点の実データ**
（出走表・モーター履歴・当地成績・asahi_config）がそろった環境が必要。
運用者のローカル（Windows）または該当データを持つ環境で1回だけ実行する。

### 1. データ供給の結線（実装済み）

`_load_race_inputs(eval_id)` は本番と同一経路で結線済み:
`notify_arashi.fetch_programs`（出走表）→ `notify_arashi._extract_boats_from_program`
（fanファイル補完込みのBoatInfo構築）→ `BOAT_ATTRS` のdict化。
race_grade は program["race_grade_number"]、is_night はナイター場集合
{4,6,12,17,20,21,22,23,24}（いずれも notify_arashi の本番ロジックと同一）。
**本番モジュールは import して呼ぶだけで、一切変更しない。**

### 1.5. 前提条件の確認

- リポジトリが `freeze-v1-baseline` の状態であること（厳守ルール1）
- **fanファイルがfreeze期間（2026前期相当）と同一期のものであること**。
  出走表（BoatraceOpenAPI）は日付指定の静的データだが、fanファイルは期別データのため、
  期が替わった後に生成するとboatsのコース別属性が変わり、Goldenが再現不能になる。

### 2. 実行

```
python tools/golden_wrapper.py \
    --candidates tests/regression/golden/golden_100_candidates.json \
    --out tests/regression/golden4 \
    --source-commit <freeze時点のgit rev>   # 省略時は現在のHEADを自動記録（Should）
```

`golden_100_candidates.json` の eval_id 100件（Step2の出力側Goldenと同一集合＝C6）が対象。

### 3. 生成物

```
tests/regression/golden4/
├── inputs/{eval_id}.json      # build_race_evaluation_v4 への入力（100件）
├── expected/{eval_id}.json    # 同関数の出力＝期待値（100件）
└── manifest.json              # 各JSONのSHA-256、件数、source_commit、baseline_tag
```

### 4. 固定（コミット）

生成物一式をリポジトリへコミットする。以後の回帰テスト
（`tests/regression/test_ver4_golden.py`）は Golden JSON のみで完結し、実データ不要。

## 回帰テストの挙動

- Golden 未生成（manifest.json 不在）の環境では、回帰テストは **skip** される。
- Golden 生成済みなら、件数100件（C6）とSHA-256一致（C7）を検証する。
- Ver4Engine との入力→出力一致比較は **Step4-5** で本テストに追加する
  （`test_ver4_golden.py` 内のコメント位置）。
