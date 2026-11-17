"""正式G3（B案）テストの共通部品。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional
from unittest import mock

from models.evaluation import FeatureSet, Prediction, RaceEvaluation

REBUILD_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = REBUILD_DIR.parent
# 絶対パス（N-01 の境界は作業ディレクトリを一時フォルダに変えるため）
GOLDEN_DIR = REBUILD_DIR / "tests/regression/golden_buyscore"
MANIFEST = GOLDEN_DIR / "manifest.json"
CONFIG_FIXTURE = GOLDEN_DIR / "buyscore_config_freeze.json"
CONFIG_SHA256 = "da6a4edaf8220aa52ed3e4577c23ff443509512a971d0ba322079eb4c4cd6f1d"

SEED_EQUIV = 20261112
SEED_LEGACY = 20261113
N_RANDOM = 10000

LEGACY_RACE_TYPES = ("4カド型", "イン逃げ型", "1残り荒れ型", "大荒れ型", "モンスター型", "混戦型")
BUYSCORE_RACE_TYPES = ("本命戦", "混戦", "超混戦", "荒れ戦")
RACE_TYPES = LEGACY_RACE_TYPES + BUYSCORE_RACE_TYPES + ("",)


# N-01：トークンがある環境の試験用。実在しないリポジトリ名と、偽のトークン（値に意味は無い）。
# 実際のトークンが環境にあっても、試験の間はこの値で上書きされ、境界の中では外される。
FAKE_TOKEN_ENV = {
    "GITHUB_TOKEN": "ghp_G3FAKE0000000000000000000000000000",
    "GH_TOKEN": "gho_G3FAKE0000000000000000000000000000",
    "GITHUB_REPOSITORY": "g3-invalid-owner/g3-invalid-repo",
}


def fake_token_env():
    return mock.patch.dict(os.environ, FAKE_TOKEN_ENV)


def config_bytes(cfg: Mapping[str, Any]) -> bytes:
    return json.dumps(cfg, ensure_ascii=False).encode("utf-8")


def iso_factory(seeds: Optional[Mapping[str, Any]] = None):
    """既定は freeze 版の設定（golden の固定ファイル）だけを作業フォルダに置く境界。"""
    from shadow.g3_isolation import G3Isolation

    use = {"buyscore_config.json": CONFIG_FIXTURE} if seeds is None else dict(seeds)
    return lambda: G3Isolation(use)


def load_freeze_config() -> dict:
    data = CONFIG_FIXTURE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CONFIG_SHA256:
        raise AssertionError(f"freeze config modified: {digest}")
    return json.loads(data.decode("utf-8"))


def verify_golden_manifest() -> list[tuple[dict, dict]]:
    """manifest の SHA-256 と入力・期待値ファイルの一致を確かめ、(input, expected) を返す。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pairs = []
    for e in manifest["entries"]:
        ip = GOLDEN_DIR / "inputs" / f"{e['idx']}.json"
        xp = GOLDEN_DIR / "expected" / f"{e['idx']}.json"
        if hashlib.sha256(ip.read_bytes()).hexdigest() != e["input_sha256"]:
            raise AssertionError(f"golden input sha256 mismatch: {ip}")
        if hashlib.sha256(xp.read_bytes()).hexdigest() != e["expected_sha256"]:
            raise AssertionError(f"golden expected sha256 mismatch: {xp}")
        pairs.append((json.loads(ip.read_text(encoding="utf-8")),
                      json.loads(xp.read_text(encoding="utf-8"))))
    if len(pairs) != manifest["count"]:
        raise AssertionError("golden count mismatch")
    return pairs


def evaluation(**over: Any) -> RaceEvaluation:
    base = dict(
        eval_id="20261110_12_05", race_date="20261110", venue_num=12,
        venue_name="住之江", race_number=5, is_night=False,
        engine_name="ver4", engine_version="4.0.0", feature_schema_version=1,
        model_version="m", evaluated_at="t",
        features=FeatureSet(
            eval_id="20261110_12_05", feature_schema_version=1, built_at="t",
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        ),
        danger_score=10.0, danger_breakdown={}, upset_score=5.0, upset_reasons=(),
        rank_index={}, featured_boats=None, win_probs=None, race_type="",
        match_index=52.5,
    )
    base.update(over)
    return RaceEvaluation(**base)


def prediction(**over: Any) -> Prediction:
    base = dict(
        eval_id="20261110_12_05", pred_combo="1-2-3", pred_prob=0.06,
        pred_ev=2.0, pred_odds=35.0, confidence=0.5, why_bet="x", patterns=(),
    )
    base.update(over)
    return Prediction(**base)


def random_equiv_case(rng: random.Random) -> dict:
    """入力の追加なしの同値テスト用（生成の順序を固定）。"""
    mi_choice = rng.random()
    match_index = None if mi_choice < 0.02 else rng.uniform(0.0, 100.0)
    return {
        "pred_prob": rng.uniform(0.0, 1.0),
        "pred_ev": rng.uniform(0.0, 10.0),
        "pred_odds": rng.uniform(1.0, 1000.0),
        "match_index": match_index,
        "upset_score": rng.uniform(0.0, 13.5),
        "race_type": rng.choice(RACE_TYPES),
    }


def random_legacy_case(rng: random.Random) -> dict:
    """入力の追加ありの Legacy 比較用（生成の順序を固定）。"""
    return {
        "pred_prob": rng.uniform(0.0, 1.0),
        "pred_ev": rng.uniform(0.0, 10.0),
        "pred_odds": rng.uniform(1.0, 1000.0),
        "uncertainty": rng.uniform(0.0, 1.0),
        "disagreement": rng.uniform(0.0, 1.0),
        "composite": rng.uniform(0.0, 1.0),
        "has_exhibition": rng.random() < 0.5,
        "market_gap": rng.uniform(-1.0, 1.0),
        "match_index_approx": rng.random() < 0.5,
        "score_post": rng.uniform(0.0, 13.5),
        "race_type": rng.choice(RACE_TYPES),
    }


def cases_sha256(cases: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def save_failures(name: str, failures: list[dict]) -> str:
    """失敗した入力をテストの一時フォルダに保存する（リポジトリには書かない）。"""
    d = Path(os.environ.get("G3_TEST_FAILURE_DIR") or tempfile.mkdtemp(prefix="g3_fail_"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(failures[:200], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return str(p)
