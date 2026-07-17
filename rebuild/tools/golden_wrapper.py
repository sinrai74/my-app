"""
tools/golden_wrapper.py — Ver4評価のGolden Input/Output生成Wrapper（Step4-0）。

目的:
  freeze-v1-baseline時点の実データで build_race_evaluation_v4 を「外側から」呼び、
  その入力と出力の両方を保存して Golden Input/Output（各100件）を生成する。

厳守事項（Step4計画書 C1/C2/C6/C7）:
  - x_asahi_scoring.build_race_evaluation_v4 は import して呼ぶだけ。**本体は一切変更しない**。
  - 一時フックを本体へ差し込まない（このWrapperが唯一の抽出経路）。
  - 生成物: golden/inputs/{eval_id}.json, golden/expected/{eval_id}.json, golden/manifest.json
  - manifest に各JSONのSHA-256を記録し変更禁止の基準とする（C7）。

実行環境:
  本Wrapperは freeze時点の実データ（出走表・モーター履歴・当地成績・config）が
  そろった環境（運用者のローカル/Actions）で1回だけ実行する。生成後はGolden一式を
  リポジトリへ固定し、以後の回帰はGolden JSONのみで行う（実データ不要）。

使い方:
  python tools/golden_wrapper.py \
      --candidates tests/regression/golden/golden_100_candidates.json \
      --out tests/regression/golden4 \
      [--source-commit <git rev>]

  --candidates: golden_100_candidates.json（eval_id 100件と選定ルールを記録済み）
  boats等の実データの供給は _load_race_inputs() を運用者環境に合わせて実装する
  （下記 NotImplementedError 参照。ここはデータ供給の結線ポイントであり、
   評価ロジックには一切手を入れない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# build_race_evaluation_v4 は「呼ぶだけ」。本体は変更しない。
from x_asahi_scoring import build_race_evaluation_v4, load_asahi_config
from x_venue_stats import compute_venue_course_stats, get_venue_course_factor

# Golden Inputに保存するboat属性（x_asahi_scoringが参照する全属性）。
# この集合はWrapperがboatを再構築（round-trip）するための最小十分な情報。
# Wrapper自体の版数。生成物のmanifestへ記録し、生成ロジック変更の追跡を可能にする（Should対応）。
GENERATOR_VERSION: str = "1.1.0"  # v1.1.0: danger_venue_factors（レーン別場補正）を入力へ追加

BOAT_ATTRS: tuple[str, ...] = (
    "lane",
    "name",
    "racer_class",
    "win_rate",
    "avg_st",
    "motor",
    "local_win",
    "ability_curr",
    "ability_prev",
    "course_nyuko",
    "course_win_rate",
    "course_place_rate",
    "course_place_counts",
    "course_rank",
    "course_st",
    "course_f_count",
    "course_l_count",
)


@dataclass(frozen=True)
class RaceInput:
    """1レース分のGolden Input（build_race_evaluation_v4の全引数を保持）。"""

    eval_id: str
    boats: list[dict[str, Any]]  # 各boatの属性dict（BOAT_ATTRS）
    venue: Optional[str]
    race_grade: int
    venue_num: int
    is_night: bool
    config_version: str  # 使用したconfigの _version（値そのものは別途freeze記録）
    danger_venue_factors: Optional[dict] = None  # v1.1.0: 生成時のレーン別場補正
    # {"course_rentai": {"1".."6": factor}, "venue_unfavorable": factor}
    # calc_danger_score_v2の2種の呼び出し（cfg付き/なし）を各々忠実に記録する

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "boats": self.boats,
            "venue": self.venue,
            "race_grade": self.race_grade,
            "venue_num": self.venue_num,
            "is_night": self.is_night,
            "config_version": self.config_version,
            "danger_venue_factors": self.danger_venue_factors,
        }


class _BoatView:
    """dict属性をオブジェクト風（b.lane 等）に見せる薄いアダプタ。

    build_race_evaluation_v4 は boat.lane のような属性アクセスを行うため、
    Golden Inputのdictから復元する際にこのビューでラップする。
    評価ロジックには一切関与しない（属性の読み出しのみ）。
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.__dict__.update(data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _load_race_inputs(eval_id: str) -> RaceInput:
    """eval_id から RaceInput を組み立てる（本番と同一のデータ経路で結線済み）。

    boats構築は本番経路をそのまま利用する（Step4-0B結線・本体は無改変）:
      - 出走表: notify_arashi.fetch_programs(race_date)（BoatraceOpenAPI）
      - boats:  notify_arashi._extract_boats_from_program(program)
                （fanファイルによるコース別ST・進入回数等の補完を含む）
      - race_grade: program["race_grade_number"]（notify_arashi L1182と同一）
      - is_night:   venue_num in NIGHT_VENUES（notify_arashi L3448と同一集合）

    注意: 出走表は日付指定の静的データだが、fanファイルは期別データのため、
    Golden生成はfreeze期間と同一期のfanファイルが配置された環境で実行すること
    （READMEの前提条件を参照）。
    """
    # 本番モジュールは実行環境（運用者Windows）でのみimportする
    from notify_arashi import (  # 呼ぶだけ・無改変
        VENUE_NAMES,
        _extract_boats_from_program,
        fetch_programs,
    )

    date, venue_s, race_s = eval_id.split("_")
    venue_num = int(venue_s)
    race_number = int(race_s)

    programs = fetch_programs(date)
    program = next(
        (
            p for p in programs
            if int(p.get("race_stadium_number", 0)) == venue_num
            and int(p.get("race_number", 0)) == race_number
        ),
        None,
    )
    if program is None:
        raise RuntimeError(f"program not found for {eval_id}")

    boat_objs = _extract_boats_from_program(program)
    boats = [
        {attr: getattr(b, attr, None) for attr in BOAT_ATTRS} for b in boat_objs
    ]
    night_venues = {4, 6, 12, 17, 20, 21, 22, 23, 24}  # notify_arashi L3447と同一
    config = load_asahi_config()
    venue_name_str = VENUE_NAMES.get(venue_num, f"場{venue_num}")

    # v1.1.0: danger算出で使われるレーン別場補正を「生成時の値」として記録する。
    # 呼び出し形はcalc_danger_score_v2内と完全同一:
    #   course_rentai用: get_venue_course_factor(venue, lane, venue_stats, cfg.get("_venue_cfg"))
    #   venue_unfavorable用: get_venue_course_factor(venue, 1, venue_stats)  ※cfgなし
    danger_venue_factors = None
    if venue_name_str:
        venue_stats = compute_venue_course_stats()
        course_rentai_factors = {
            str(lane): get_venue_course_factor(
                venue_name_str, lane, venue_stats, config.get("_venue_cfg")
            )["factor"]
            for lane in range(1, 7)
        }
        unfavorable_factor = get_venue_course_factor(
            venue_name_str, 1, venue_stats
        )["factor"]
        danger_venue_factors = {
            "course_rentai": course_rentai_factors,
            "venue_unfavorable": unfavorable_factor,
        }

    return RaceInput(
        eval_id=eval_id,
        boats=boats,
        venue=venue_name_str,
        race_grade=int(program.get("race_grade_number", 0) or 0),
        venue_num=venue_num,
        is_night=venue_num in night_venues,
        config_version=str(config.get("_version", "")),
        danger_venue_factors=danger_venue_factors,
    )


def _boats_from_dicts(boat_dicts: list[dict[str, Any]]) -> list[_BoatView]:
    return [_BoatView(d) for d in boat_dicts]


def generate(candidates_path: Path, out_dir: Path, source_commit: Optional[str]) -> None:
    """Golden Input/Output（各100件）とmanifestを生成する。"""
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    eval_ids: list[str] = candidates["eval_ids"]

    inputs_dir = out_dir / "inputs"
    expected_dir = out_dir / "expected"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    config = load_asahi_config()
    commit = source_commit or _current_commit()

    manifest_entries: list[dict[str, Any]] = []
    for eval_id in eval_ids:
        race_input = _load_race_inputs(eval_id)

        # 入力を保存
        in_path = inputs_dir / f"{eval_id}.json"
        in_path.write_text(
            json.dumps(race_input.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        # build_race_evaluation_v4 を「外側から」呼ぶ（本体無改変）
        result = build_race_evaluation_v4(
            boats=_boats_from_dicts(race_input.boats),
            venue=race_input.venue,
            race_grade=race_input.race_grade,
            venue_num=race_input.venue_num,
            is_night=race_input.is_night,
            config=config,
        )

        # 期待出力を保存
        out_path = expected_dir / f"{eval_id}.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        manifest_entries.append(
            {
                "eval_id": eval_id,
                "input_sha256": _sha256(in_path),
                "expected_sha256": _sha256(out_path),
            }
        )

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "created_from": "build_race_evaluation_v4 (unmodified)",
        "baseline_tag": "freeze-v1-baseline",
        "source_commit": commit,  # Should: 取得元コミットID
        "count": len(manifest_entries),
        "candidates_source": str(candidates_path),
        "entries": manifest_entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"generated {len(manifest_entries)} golden input/expected pairs into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ver4 Golden Input/Output.")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    generate(args.candidates, args.out, args.source_commit)


if __name__ == "__main__":
    main()
