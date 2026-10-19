"""
x_ranking.yml の検証・実績ページ・note の3ステップが、Release に保存されたデータを読めること。

GITHUB_TOKEN が環境変数に無いと x_release_storage.is_available() が False になり、
download_file は何も読まずに戻る（実績ページ・検証が空になる）。
"""
import csv
import io
import json

import pytest
import yaml

from test_release_storage_atomic_20261018 import FakeGH, gh  # noqa: F401  (fixture)

STEPS = ["Send - Verification", "Send - Results Page", "Send Note Report"]


def _steps():
    with open(".github/workflows/x_ranking.yml", encoding="utf-8") as f:
        wf = yaml.safe_load(f)
    out = {}
    for job in wf["jobs"].values():
        for st in job.get("steps", []):
            if st.get("name") in STEPS:
                out[st["name"]] = st
    return out


@pytest.fixture
def repo_root(request):
    return request.config.rootpath


def test_three_steps_pass_github_token(monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    steps = _steps()
    assert set(steps) == set(STEPS)
    for name, st in steps.items():
        assert st["env"].get("GITHUB_TOKEN") == "${{ secrets.GITHUB_TOKEN }}", name


def test_three_steps_do_not_write_to_release(monkeypatch, repo_root):
    """トークンを渡すステップのスクリプトは、Release に書き込まない（読み込みのみ）"""
    monkeypatch.chdir(repo_root)
    import re
    scripts = set()
    for st in _steps().values():
        scripts |= set(re.findall(r"python (x_\w+\.py)", st["run"]))
    assert scripts == {"x_verification.py", "x_buyscore_tuner.py", "x_results_page.py", "x_note_report.py"}
    for st in _steps().values():
        if "x_buyscore_tuner.py" in st["run"]:
            assert "x_buyscore_tuner.py --dry-run" in st["run"]   # save_config は dry-run で呼ばれない


HDR = ["date", "venue_num", "race", "pred_combo", "result_combo", "payout", "hit"]


def _hit_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HDR)
    w.writerows(rows)
    return buf.getvalue().encode()


def test_verification_reads_saved_hit_record_with_token(gh):
    import x_verification
    gh.put("hit_record.csv", _hit_csv([
        ["20261017", "1", "1", "1-2-3", "1-2-3", "1240", "1"],
        ["20261017", "1", "1", "1-2-3", "不明", "0", "0"],     # 同じキーの2行目（先の行が使われる）
        ["20261016", "2", "3", "2-1-3", "3-1-2", "0", "0"],
    ]))
    recs = x_verification.load_today_records("20261017")
    assert [(r["race"], r["result_combo"]) for r in recs] == [("1", "1-2-3")]


def test_verification_reads_nothing_without_token(gh, monkeypatch):
    """修正前の状態の再現：トークンが無いと保存済みデータを読めない"""
    import x_verification
    gh.put("hit_record.csv", _hit_csv([["20261017", "1", "1", "1-2-3", "1-2-3", "1240", "1"]]))
    monkeypatch.delenv("GITHUB_TOKEN")
    assert x_verification.load_today_records("20261017") == []


def test_results_common_reads_buyscore_log_with_token(gh):
    import x_results_common
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
    rec = {"date": today, "venue": "丸亀", "race": "4", "passthrough_reason": None,
           "candidates": [{"combo": "1-2-3", "buyscore": 70, "amount": 100}]}
    gh.put("buyscore_log.jsonl", (json.dumps(rec, ensure_ascii=False) + "\n").encode())
    res = x_results_common.analyze_buyscore_log(days=7)
    assert res.get("data_available") is not False
