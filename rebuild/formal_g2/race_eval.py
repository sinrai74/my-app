"""正式G2：会場統計の事前読み込みと、1レースの評価・比較（判断A・判断C C-0・判断D）。

1レースあたり：resolve_race 1回、resolve_boats 1回（戻り値を L とする）、
Ver4Engine.evaluate 1回、凍結版 build_race_evaluation_v4 1回、compare 1回。
加算・連続数は判定しない（記録だけ。集約処理で決める）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Any, Callable, Optional

JST = timezone(timedelta(hours=9))
FIVE_ITEMS = ("danger_score", "danger_breakdown", "upset_score", "rank_index", "featured_boats")
COMPARATOR_ALIAS = "formal_g2_comparator"
NATIONAL_KEY = "全国平均"
LANES = (1, 2, 3, 4, 5, 6)


# ---- ハッシュ（判断D 2・3・4） ----
def stable_hash(obj: Any) -> str:
    """判断D 2 の方法。文字列にできなければ 'UNSERIALIZABLE:…' を返す（判断D 4）。"""
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=True, separators=(",", ":"))
    except Exception as exc:  # noqa: BLE001
        return f"UNSERIALIZABLE:{type(exc).__name__}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ok_hash(h: Optional[str]) -> bool:
    return isinstance(h, str) and not h.startswith("UNSERIALIZABLE")


def comparable_hash(to_comparable: Callable[[Any], Any], obj: Any) -> str:
    """判断D 11(6)②：比較器と同じ文字列化（ensure_ascii=False, sort_keys=True）。"""
    try:
        s = json.dumps(to_comparable(obj), ensure_ascii=False, sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        return f"UNSERIALIZABLE:{type(exc).__name__}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _file_sha(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


# ---- comparator の直接読み込み（shadow/__init__.py を実行しない） ----
def load_comparator(repo_root: str) -> ModuleType:
    path = os.path.join(repo_root, "rebuild", "shadow", "comparator.py")
    spec = importlib.util.spec_from_file_location(COMPARATOR_ALIAS, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[COMPARATOR_ALIAS] = module
    spec.loader.exec_module(module)  # 依存が解決できなければ例外のまま止める
    return module


def shadow_package_loaded() -> list[str]:
    return sorted(n for n in sys.modules if n == "shadow" or n.startswith("shadow."))


# ---- 会場統計・会場設定の事前読み込み（判断D 5、評価開始前に必ず実行） ----
class PreloadStop(Exception):
    def __init__(self, reason: str, record: dict) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record = record


def validate_venue_stats(stats: Any) -> list[str]:
    """空・全国平均の 1〜6 コースの欠落・samples が 0 以下を問題として返す。"""
    problems = []
    if not isinstance(stats, dict) or not stats:
        return ["venue stats empty"]
    national = stats.get(NATIONAL_KEY)
    if not isinstance(national, dict) or not national:
        return ["national average missing"]
    for lane in LANES:
        cell = national.get(lane)
        if not isinstance(cell, dict):
            problems.append(f"national lane {lane} missing")
            continue
        samples = cell.get("samples")
        if not isinstance(samples, (int, float)) or samples <= 0:
            problems.append(f"national lane {lane} samples <= 0")
    return problems


def preload_venue_caches(repo_root: str, race_date: str) -> dict[str, Any]:
    """凍結版の関数を既定の引数で呼び、キャッシュを作って検証・記録する。失敗は PreloadStop。"""
    import x_venue_stats

    rec: dict[str, Any] = {"race_date": race_date, "ok": False}
    rec["loaded_before"] = {
        "venue_stats": x_venue_stats._VENUE_STATS_CACHE is not None,
        "venue_config": x_venue_stats._VENUE_CONFIG_CACHE is not None,
    }
    history = os.path.join(repo_root, x_venue_stats.MOTOR_HISTORY_FILE)
    rec["source"] = {
        "motor_history_file": x_venue_stats.MOTOR_HISTORY_FILE,
        "motor_history_sha256": _file_sha(history),
        "venue_config_exists": os.path.isfile(os.path.join(repo_root, "venue_config.json")),
        "venue_config_sha256": _file_sha(os.path.join(repo_root, "venue_config.json")),
        "note": "venue stats do not depend on race_date; computed from the checked-out motor_history.csv",
    }
    try:
        stats = x_venue_stats.compute_venue_course_stats()
        vcfg = x_venue_stats.load_venue_config()
    except Exception as exc:  # noqa: BLE001
        rec["exception"] = f"{type(exc).__name__}: {exc}"
        raise PreloadStop("preload raised an exception", rec) from exc
    rec["loaded_at"] = datetime.now(JST).isoformat()
    rec["venue_stats_sha256"] = stable_hash(x_venue_stats._VENUE_STATS_CACHE)
    rec["venue_stats_id"] = id(x_venue_stats._VENUE_STATS_CACHE)
    rec["venue_config_sha256"] = stable_hash(x_venue_stats._VENUE_CONFIG_CACHE)
    rec["venue_config_id"] = id(x_venue_stats._VENUE_CONFIG_CACHE)
    rec["returned_is_cache"] = (stats is x_venue_stats._VENUE_STATS_CACHE
                                and vcfg is x_venue_stats._VENUE_CONFIG_CACHE)
    rec["venue_count"] = max(len(stats) - 1, 0) if isinstance(stats, dict) else None
    national = stats.get(NATIONAL_KEY, {}) if isinstance(stats, dict) else {}
    rec["national_samples"] = {
        str(lane): (national.get(lane) or {}).get("samples") for lane in LANES
    }
    problems = validate_venue_stats(stats)
    if rec["source"]["motor_history_sha256"] is None:
        problems.append("motor_history.csv unreadable")
    for key in ("venue_stats_sha256", "venue_config_sha256"):
        if not _ok_hash(rec[key]):
            problems.append(f"{key} unavailable")
    if not rec["returned_is_cache"]:
        problems.append("returned objects are not the module caches")
    rec["problems"] = problems
    if problems:
        raise PreloadStop("preload validation failed", rec)
    rec["ok"] = True
    return rec


# ---- 実行全体で共有する文脈 ----
@dataclasses.dataclass
class RunContext:
    repo_root: str
    race_date: str
    programs: list[dict]
    cfg: dict
    comparator: ModuleType
    instrument: Any
    provider: Any = None
    fetcher: Any = None
    preload: Optional[dict] = None


def build_context(repo_root: str, race_date: str, programs: list[dict], cfg: dict,
                  comparator: ModuleType, instrument: Any, preload: dict) -> RunContext:
    from adapters.providers import BoatsProvider

    from formal_g2.programs import FixedFetcher

    fetcher = FixedFetcher(race_date, programs)
    provider = BoatsProvider(programs_fetcher=fetcher)
    return RunContext(repo_root, race_date, programs, cfg, comparator, instrument, provider, fetcher, preload)


def _cache_hashes(ctx: RunContext, L: list, legacy_vars: Optional[list]) -> dict[str, Any]:
    import notify_arashi
    import x_venue_stats

    out = {
        "b_L": stable_hash(L),
        "d_cfg": stable_hash(ctx.cfg),
        "e_venue_stats": stable_hash(x_venue_stats._VENUE_STATS_CACHE),
        "e_venue_stats_id": id(x_venue_stats._VENUE_STATS_CACHE),
        "e_venue_config": stable_hash(x_venue_stats._VENUE_CONFIG_CACHE),
        "e_venue_config_id": id(x_venue_stats._VENUE_CONFIG_CACHE),
        "f_fan_cache": stable_hash(notify_arashi._FAN_CACHE),
        "f_fan_file": notify_arashi._get_fan_file(),
    }
    if legacy_vars is not None:
        out["c_legacy_boats"] = stable_hash(legacy_vars)
    return out


def _matches_preload(ctx: RunContext, h: dict) -> bool:
    p = ctx.preload or {}
    return (
        h["e_venue_stats"] == p.get("venue_stats_sha256")
        and h["e_venue_stats_id"] == p.get("venue_stats_id")
        and h["e_venue_config"] == p.get("venue_config_sha256")
        and h["e_venue_config_id"] == p.get("venue_config_id")
    )


def _read_provider_cache(provider: Any, key: tuple) -> tuple[Any, dict]:
    """provider._cache を読むだけ。読む前後で型・件数・キー集合が変わらないことを記録する。"""
    cache = provider._cache

    def shape() -> dict:
        return {"type": type(cache).__name__, "len": len(cache), "keys": stable_hash(sorted(map(repr, cache)))}

    before = shape()
    value = cache[key]
    after = shape()
    return value, {"before": before, "after": after, "unchanged": before == after,
                   "same_object": cache is provider._cache}


def evaluate_race(ctx: RunContext, target: dict) -> dict[str, Any]:
    """1レースを評価・比較し、レースごとの記録（判断D 11(3)・D5）を返す。"""
    import notify_arashi
    from adapters.providers import BOAT_ATTRS
    from core.engine import Ver4Engine, _grade_to_number
    from models.evaluation import FeatureSet
    from tools.golden_wrapper import _BoatView
    from x_asahi_scoring import build_race_evaluation_v4
    import x_venue_stats

    eval_id = target["eval_id"]
    venue_num = target["venue_num"]
    race_number = target["race_number"]
    rec: dict[str, Any] = {"eval_id": eval_id, "stage": None, "exception": None, "config_detected": None}
    counts_before = dict(ctx.instrument.counts)
    fetch_calls_before = len(ctx.fetcher.calls)

    def close(category: str, states: list[str]) -> dict[str, Any]:
        rec["result_category"] = category
        rec["d9_states"] = states
        rec["input_identity_ok"] = False
        _finish_counts(ctx, rec, counts_before, fetch_calls_before)
        rec["d5"] = _d5(rec)
        return rec

    def fail(stage: str, exc: BaseException) -> dict[str, Any]:
        rec["stage"] = stage
        rec["exception"] = {"class": type(exc).__name__, "message": str(exc)[:500], "stage": stage}
        cat = "compare_exception" if stage == "compare" else "not_established"
        return close(cat, ["compare_not_established"])

    def config_hit(step: str) -> bool:
        """区切りで config の実行・読み込みを確かめる。検出したらこのレースは打ち切る。"""
        if ctx.instrument.config_executed or "config" in sys.modules:
            rec["config_detected"] = {"at_step": step, "phase": ctx.instrument.phase,
                                      "detections": list(ctx.instrument.detections)}
            return True
        return False

    # 1. 入力の解決（L を得る）
    try:
        race = ctx.provider.resolve_race(ctx.race_date, venue_num, race_number)
        L = list(ctx.provider.resolve_boats(ctx.race_date, venue_num, race_number))
    except Exception as exc:  # noqa: BLE001
        return fail("resolve", exc)
    if config_hit("after_resolve"):
        return close("not_established", ["compare_not_established"])
    (program, boat_objs), cache_check = _read_provider_cache(
        ctx.provider, (ctx.race_date, venue_num, race_number)
    )
    rec["provider_cache_read_check"] = cache_check
    a_boatinfo = [dataclasses.asdict(b) for b in boat_objs]
    rec["program_sha256"] = stable_hash(program)
    rec["a_boatinfo"] = stable_hash(a_boatinfo)
    rec["race"] = {
        "venue_num": race.venue_num, "venue_name": race.venue_name, "is_night": race.is_night,
        "grade": race.grade, "race_grade_from_race": _grade_to_number(race.grade),
        "race_grade_number_in_program": program.get("race_grade_number"),
    }
    rec["none_attributes_by_boat"] = {
        str(row.get("lane")): sorted(k for k in BOAT_ATTRS if row.get(k) is None) for row in L
    }

    # 2. Legacy 側の形（_BoatView）と (b)/(c) の一致（判断D 6）
    boats_view = [_BoatView(dict(row)) for row in L]
    c_vars = [vars(v) for v in boats_view]
    per_attr_match = len(L) == len(c_vars) and all(
        stable_hash(row.get(k)) == stable_hash(cv.get(k)) for row, cv in zip(L, c_vars) for k in BOAT_ATTRS
    )
    rec["b_c_match"] = per_attr_match
    venue_str = notify_arashi.VENUE_NAMES.get(race.venue_num)
    rec["venue_for_legacy"] = venue_str
    race_grade_num = _grade_to_number(race.grade)

    # 3. 評価前のハッシュ
    h_before = _cache_hashes(ctx, L, c_vars)
    h_before["a_boatinfo"] = rec["a_boatinfo"]

    # 4. Rebuild の評価（Ver4Engine.evaluate 1回）
    feature_set = FeatureSet(
        eval_id=eval_id, feature_schema_version=1,
        built_at=datetime.now(JST).isoformat(),
        boat_features={1: {}}, race_features={}, local_features=None, missing_keys=(),
    )
    engine = Ver4Engine(boats_resolver=lambda _race: L, venue_stats=x_venue_stats, predictor=None)
    try:
        evaluation = engine.evaluate(race, feature_set, None, ctx.cfg, datetime.now(JST))
    except Exception as exc:  # noqa: BLE001
        return fail("rebuild_evaluate", exc)
    if config_hit("after_rebuild_evaluate"):
        return close("not_established", ["compare_not_established"])
    h_after_rebuild = _cache_hashes(ctx, L, c_vars)

    # 5. 凍結版 v4（1回）
    try:
        legacy = build_race_evaluation_v4(
            boats=boats_view, venue=venue_str, race_grade=race_grade_num,
            venue_num=race.venue_num, is_night=race.is_night, config=ctx.cfg,
        )
    except Exception as exc:  # noqa: BLE001
        return fail("legacy_v4", exc)
    if config_hit("after_legacy_v4"):
        return close("not_established", ["compare_not_established"])
    h_after_v4 = _cache_hashes(ctx, L, [vars(v) for v in boats_view])
    h_after_v4["a_boatinfo"] = stable_hash([dataclasses.asdict(b) for b in boat_objs])

    # 6. 比較（判断A：5項目、featured_boats は Legacy 側を包む）
    legacy_values = {k: legacy[k] for k in FIVE_ITEMS if k != "featured_boats"}
    legacy_values["featured_boats"] = {"featured": legacy["featured_boats"]}
    rec["legacy_empty_dict_items"] = sorted(k for k, v in legacy_values.items() if v == {})
    try:
        diffs = ctx.comparator.compare(eval_id, legacy_values, evaluation)
    except Exception as exc:  # noqa: BLE001
        return fail("compare", exc)
    if config_hit("after_compare"):
        return close("not_established", ["compare_not_established"])

    to_cmp = ctx.comparator.to_comparable
    try:
        rebuild_c = to_cmp(evaluation)
        rebuild_five = {k: rebuild_c.get(k) for k in FIVE_ITEMS}
    except Exception as exc:  # noqa: BLE001
        rebuild_five = None
        rec["rebuild_values_error"] = f"{type(exc).__name__}: {exc}"
    rec["values"] = {"legacy": legacy_values, "rebuild": rebuild_five}
    rec["comparable_hash"] = {
        "legacy": comparable_hash(to_cmp, legacy_values),
        "rebuild": comparable_hash(to_cmp, rebuild_five) if rebuild_five is not None else None,
    }
    rec["result_category"] = "diff_zero" if not diffs else "mismatch"
    rec["diffs"] = diffs
    rec["meta_check"] = {
        "venue_num": evaluation.venue_num == race.venue_num,
        "venue_name": evaluation.venue_name == race.venue_name,
        "is_night": evaluation.is_night == race.is_night,
        "model_version": evaluation.model_version == legacy.get("model_version"),
        "venue_matches_VENUE_NAMES": venue_str == notify_arashi.VENUE_NAMES.get(race.venue_num),
        "race_grade_program_vs_race": int(program.get("race_grade_number", 0) or 0) == race_grade_num,
    }
    rec["hashes"] = {"before": h_before, "after_rebuild": h_after_rebuild, "after_v4": h_after_v4}
    _finish_counts(ctx, rec, counts_before, fetch_calls_before)

    # 7. 判断D 13-2 の観測条件
    #   (b)(d)(e)(f) は3時点すべてで一致、(a)(c) は評価前と v4 の後で一致（判断D 6）、
    #   (e) は3時点すべてで事前読み込みの値（ハッシュと id）と一致すること。
    stable_keys = ("b_L", "d_cfg", "e_venue_stats", "e_venue_config", "f_fan_cache")
    unstable = [k for k in stable_keys if not (h_before[k] == h_after_rebuild[k] == h_after_v4[k])]
    unstable += [k for k in ("a_boatinfo", "c_legacy_boats") if h_before[k] != h_after_v4[k]]
    rec["unstable_hash_keys"] = unstable
    preload_match = {
        "before": _matches_preload(ctx, h_before),
        "after_rebuild": _matches_preload(ctx, h_after_rebuild),
        "after_v4": _matches_preload(ctx, h_after_v4),
    }
    rec["d13_2"] = {
        "b_c_match": per_attr_match,
        "hashes_stable": not unstable,
        "venue_caches_match_preload": all(preload_match.values()),
        "fetcher_calls_eq_1": rec["counts"]["fetcher_calls"] == 1,
        "resolve_boats_calls_eq_1": rec["counts"]["BoatsProvider.resolve_boats"] == 1,
        "venue_match": rec["meta_check"]["venue_matches_VENUE_NAMES"],
        "race_grade_match": rec["meta_check"]["race_grade_program_vs_race"],
        "provider_cache_unchanged": cache_check["unchanged"],
        "config_not_executed": not ctx.instrument.config_executed,
    }
    rec["preload_match"] = preload_match
    unserializable = any(
        isinstance(v, str) and v.startswith("UNSERIALIZABLE")
        for h in (h_before, h_after_rebuild, h_after_v4) for v in h.values()
    )
    states = []
    if not all(rec["d13_2"].values()):
        states.append("compare_not_established")  # 入力が同じでないことが観測された（判断D 9(1)）
    if unserializable:
        states.append("input_identity_unverified")  # 判断D 9(2)
    rec["d9_states"] = states
    rec["input_identity_ok"] = not states
    rec["d5"] = _d5(rec)
    return rec


def _finish_counts(ctx: RunContext, rec: dict, counts_before: dict, fetch_calls_before: int) -> None:
    now = ctx.instrument.counts
    counts = {k: now[k] - counts_before.get(k, 0) for k in now}
    counts["fetcher_calls"] = len(ctx.fetcher.calls) - fetch_calls_before
    rec["counts"] = counts
    # 禁止対象の呼び出し（W7・C-5・通知・Release 書き込み）。1回でもあればこのレースは加算しない。
    from formal_g2.instrument import FORBIDDEN_LABELS
    called = {k: v for k, v in counts.items() if k in FORBIDDEN_LABELS and v}
    rec["w7"] = {"forbidden_calls": called, "forbidden_total": sum(called.values()),
                 "labels_checked": len(FORBIDDEN_LABELS), "ok": not called}


def _d5(rec: dict) -> dict[str, Any]:
    """D5 の5項目。加算・連続数への影響は集約処理で決める（§6.3〜§6.5 の 8）。"""
    return {
        "result_category": rec.get("result_category"),
        "conditions": {
            "diff_zero": rec.get("result_category") == "diff_zero",
            "input_identity_d13_2": rec.get("input_identity_ok"),
            "m7_baseline": "run_level",
            "evidence_s6_6": "decided_by_aggregate",
        },
        "added": "decided_by_aggregate",
        "streak_effect": "decided_by_aggregate",
        "dispute_ref": None,
    }
