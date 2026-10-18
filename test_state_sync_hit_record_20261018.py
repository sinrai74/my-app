"""
sent ファイルの保存・復元（x_state_sync）と、変更していない notify_arashi._check_yesterday_results
を組み合わせた模擬実行。各実行は新しい空のディレクトリ（＝使い捨ての実行環境）で行う。
GitHub Releases は偽サーバ、レース結果の取得は固定値に置き換える。ネットワークには接続しない。
"""
import csv
import io
import json
import os

import pytest

import notify_arashi
import x_release_storage as xrs
import x_state_sync
from test_release_storage_atomic_20261018 import FakeGH, Crash, gh  # noqa: F401

D1, D2 = "20261020", "20261021"

RESULTS = {  # (venue_num, race) -> 結果
    (1, 1): {"combo": "1-2-3", "payout": 1200},
    (1, 2): {"combo": "4-5-6", "payout": 35000},
    (2, 7): {"combo": "2-1-3", "payout": 800},
}


def entry(venue, race, combo, purchased=True, ex=False):
    key = f"{D1}_{venue}_{race}" + ("_ex" if ex else "")
    return {"key": key, "combo": combo, "buy": [combo], "buy_amounts": [100],
            "prob": 0.1, "ev": 1.2, "odds": 12.0, "purchased": purchased}


@pytest.fixture
def env(gh, monkeypatch, tmp_path):
    calls = {"fetch": 0}

    def fake_fetch(venue_num, race_number, race_date):
        calls["fetch"] += 1
        return RESULTS.get((venue_num, race_number))

    monkeypatch.setattr(notify_arashi, "_fetch_race_result", fake_fetch)
    monkeypatch.setattr(notify_arashi, "_run_calibration_check", lambda *a, **k: None)
    state = {"n": 0}

    def run(date, body=None):
        """1回の Actions 実行：restore → notify（結果照合＋当日の sent 書き込み）→ save"""
        state["n"] += 1
        d = tmp_path / f"run{state['n']}"
        d.mkdir()
        monkeypatch.chdir(d)
        x_state_sync.main(["restore", "--date", date])
        notify_arashi._check_yesterday_results(date)
        if body is not None:
            with open(f"sent_{date}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in body))
        x_state_sync.main(["save"])
        return d

    return gh, run, calls


def hit_rows(gh):
    data = gh.get_data("hit_record.csv")
    return list(csv.DictReader(io.StringIO(data.decode("utf-8")))) if data else []


def test_two_consecutive_runs_record_hit_record(env):
    gh, run, calls = env
    run(D1, [entry(1, 1, "1-2-3"), entry(1, 2, "1-2-3", purchased=False)])
    assert gh.get_data(f"sent_{D1}.txt") is not None
    assert gh.get_data("hit_record.csv") is None          # 前日分が無いので記録なし

    run(D2)
    rows = hit_rows(gh)
    assert [(r["date"], r["venue_num"], r["race"], r["hit"]) for r in rows] == [
        (D1, "1", "1", "1"), (D1, "1", "2", "0")]
    assert len(gh_header(gh)) == 44                         # 現行の44列スキーマ
    assert calls["fetch"] == 2


def gh_header(gh):
    return next(csv.reader(io.StringIO(gh.get_data("hit_record.csv").decode())))


def test_later_runs_do_not_refetch_or_duplicate(env):
    gh, run, calls = env
    run(D1, [entry(1, 1, "1-2-3"), entry(1, 2, "1-2-3")])
    run(D2)
    n = calls["fetch"]
    run(D2)           # 同じ日の2回目以降
    run(D2)
    assert calls["fetch"] == n                     # 記録済みなので取り戻さず、結果も取り直さない
    assert len(hit_rows(gh)) == 2


def test_same_day_runs_accumulate_sent_first_wins(env):
    gh, run, calls = env
    run(D1, [entry(1, 1, "1-2-3")])
    run(D1, [entry(1, 1, "6-5-4"), entry(2, 7, "2-1-3")])   # 別の実行：同じ key は先の内容を残す
    lines = gh.get_data(f"sent_{D1}.txt").decode().split("\n")
    got = [(json.loads(x)["key"], json.loads(x)["combo"]) for x in lines]
    assert got == [(f"{D1}_1_1", "1-2-3"), (f"{D1}_2_7", "2-1-3")]
    run(D2)
    assert {(r["race"], r["hit"]) for r in hit_rows(gh)} == {("1", "1"), ("7", "1")}


def test_existing_release_hit_record_is_not_overwritten(env):
    gh, run, calls = env
    # Release 上に列構成の違う版がある場合、notify はローカルでそこへ44列の行を足すが、
    # 列数の合わない内容は Release に保存されない
    run(D1, [entry(1, 1, "1-2-3")])
    gh.put("hit_record.csv", b"date,venue_num,race\r\n20261001,1,1\r\n")
    run(D2)
    assert gh.get_data("hit_record.csv") == b"date,venue_num,race\r\n20261001,1,1\r\n"


def test_save_failure_is_retried_next_run(env):
    gh, run, calls = env
    gh.fault("POST", r"name=sent_\d+\.txt\.new", "error")
    run(D1, [entry(1, 1, "1-2-3")])
    assert gh.get_data(f"sent_{D1}.txt") is None
    run(D1, [entry(1, 2, "1-2-3")])                      # 次の実行は保存できる
    keys = [json.loads(x)["key"] for x in gh.get_data(f"sent_{D1}.txt").decode().split("\n")]
    assert keys == [f"{D1}_1_2"]                          # 失敗した実行の分は失われる（既知の制約）


def test_crash_during_sent_save_then_recovers(env):
    gh, run, calls = env
    run(D1, [entry(1, 1, "1-2-3")])
    gh.fault("PATCH", r"/releases/assets/\d+$", "crash")
    with pytest.raises(Crash):
        run(D1, [entry(1, 2, "1-2-3")])
    run(D2)
    assert {r["race"] for r in hit_rows(gh)} >= {"1"}


def test_marker_only_sent_file_is_not_restored(env):
    gh, run, calls = env
    gh.put(f"sent_{D1}.txt", json.dumps({"key": "__no_bets_evaluated__", "checked_races": 10}).encode())
    d = run(D2)
    assert not os.path.exists(d / f"sent_{D1}.txt")
    assert calls["fetch"] == 0


def test_no_release_at_all_first_run(env):
    gh, run, calls = env
    run(D2)
    assert gh.names() == []


def test_restore_failure_does_not_stop_run(env, monkeypatch):
    gh, run, calls = env
    run(D1, [entry(1, 1, "1-2-3")])
    gh.fault("GET", r"/releases/tags/data-store", "neterr", times=1)
    run(D2)                       # restore は失敗（警告のみ）
    run(D2)                       # 次の実行で記録される
    assert len(hit_rows(gh)) == 1


def test_unknown_result_row_is_written_once(env):
    """結果が取得できないレースは「不明」で記録され、以後は更新されない（notify の既存動作）"""
    gh, run, calls = env
    run(D1, [entry(3, 9, "1-2-3")])
    run(D2)
    rows = hit_rows(gh)
    assert [(r["race"], r["result_combo"]) for r in rows] == [("9", "不明")]
