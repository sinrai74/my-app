"""
購入ログ：保存の統合（x_release_storage）と、1レース1件化の候補（x_buyscore_canonical）の検証。
"""
import json

import pytest

import x_buyscore_canonical as canon
import x_release_storage as xrs
from test_release_storage_atomic_20261018 import FakeGH, Crash, gh  # noqa: F401  (fixture)


def rec(race, reason, n, venue="丸亀", date="20261014"):
    return {"date": date, "venue": venue, "race": race, "passthrough_reason": reason, "n": n}


E3 = [  # 2026-10-14 丸亀4R（incident E3）：1回目は見送り、2回目は通過
    rec("4", "BuyScore不足(56)", 1),
    rec("5", None, 2),
    rec("4", None, 3),
]


def test_policies_differ_on_incident_e3():
    assert [r["n"] for r in canon.select(E3, "first")] == [1, 2]
    assert [r["n"] for r in canon.select(E3, "last")] == [3, 2]
    assert [r["n"] for r in canon.select(E3, "last_pass")] == [3, 2]


def test_last_vs_last_pass_differ_when_flip_back():
    g = [rec("4", None, 1), rec("4", "BuyScore不足(40)", 2)]  # 通過 → 後で見送り
    assert canon.select(g, "last")[0]["n"] == 2
    assert canon.select(g, "last_pass")[0]["n"] == 1


def test_summary_counts_match_current_analyzer_without_dedupe():
    s = canon.summarize(E3)
    assert s["raw_records"] == 3 and s["races"] == 2
    # 現在の analyze_buyscore_log は重複を含めて数える（3件・購入2）
    assert s["first"] == {"n_races": 2, "buy_count": 1, "passthrough_count": 1}
    assert s["last"] == {"n_races": 2, "buy_count": 2, "passthrough_count": 0}


def test_rerun_appended_later_wins_under_last():
    g = [rec("1", "x", 1), rec("1", None, 2), rec("1", "x", 3)]  # 3 = 再実行
    assert canon.select(g, "last")[0]["n"] == 3


def test_partial_line_is_excluded_and_counted():
    lines = [json.dumps(rec("1", None, 1)), '{"date":"20261014","venue":"丸', ""]
    recs, bad = canon.read_jsonl(lines)
    assert len(recs) == 1 and bad == 1


def test_same_venue_race_on_other_date_is_separate():
    g = [rec("1", None, 1, date="20261014"), rec("1", None, 2, date="20261015")]
    assert len(canon.select(g, "last")) == 2


def test_two_runs_through_storage_keep_both_evaluations_in_order(gh):
    """2回の実行が同じレースを評価 → どちらの記録も失われず、保存順に並ぶ"""
    LOG = "buyscore_log.jsonl"
    for n, reason in ((1, "BuyScore不足(56)"), (2, None)):
        xrs.download_file(LOG, LOG)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec("4", reason, n), ensure_ascii=False) + "\n")
        assert xrs.upload_file(LOG, LOG)
    recs, bad = canon.read_jsonl(gh.get_data(LOG).decode().splitlines())
    assert bad == 0 and [r["n"] for r in recs] == [1, 2]
    assert canon.select(recs, "last")[0]["n"] == 2


def test_crashed_run_record_is_absent_not_partial(gh):
    """保存の途中で実行が止まった場合、その記録は「無い」か「全部ある」のどちらか"""
    LOG = "buyscore_log.jsonl"
    gh.put(LOG, (json.dumps(rec("4", None, 1)) + "\n").encode())
    xrs.download_file(LOG, LOG)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec("4", None, 2)) + "\n")
    gh.fault("POST", r"name=buyscore_log\.jsonl$", "crash")   # 本体を消した後、上げ直す前に停止
    with pytest.raises(Crash):
        xrs.upload_file(LOG, LOG)
    assert xrs.download_file(LOG, "fresh.jsonl")                # 本体が無いので .bak を読む
    recs, bad = canon.read_jsonl(open("fresh.jsonl", encoding="utf-8").read().splitlines())
    assert bad == 0 and [r["n"] for r in recs] == [1]
