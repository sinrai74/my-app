"""
新聞（x_note_report）の E1b（万舟警報の一覧の撤去）と J4（行き先の無いリンクを外す）の検証。
履歴の129版に現れない条件（万舟0件・危険艇0件・万舟だけのレースがランキングに入る日）を、
作ったデータで確かめる。Python 3.12 で実行する（x_ranking.yml と同じ）。
"""
import re

import pytest

import x_note_report as n


def danger(venue, race, score):
    return {"venue": venue, "venue_num": 1, "race": race, "score": score, "racer": f"選手{race}",
            "racer_class": "A1", "reason": "理由", "breakdown": {}}


def manshuu(venue, race, score):
    return {"venue": venue, "venue_num": 1, "race": race, "score": score,
            "key_racer": f"注目{race}", "key_reason": "理由A / 理由B"}


def data(d, m):
    return {"date": "20261018", "all_danger": d, "all_manshuu": m,
            "hot_motor": [], "awakening_motor": []}


@pytest.fixture
def gen(tmp_path, monkeypatch):
    def _gen(dat, show, link):
        monkeypatch.setattr(n, "NEWSPAPER_SHOW_MANSHUU_LIST", show)
        monkeypatch.setattr(n, "NEWSPAPER_LINK_ONLY_EXISTING", link)
        p = tmp_path / f"o_{show}_{link}.html"
        n.generate_html(dat, str(p))
        return p.read_text(encoding="utf-8")
    return _gen


def dangling(h):
    ids = set(re.findall(r'\bid="([^"]+)"', h))
    return [t for t in re.findall(r'<a href="#([^"]*)"', h) if t not in ids]


def section(h, sid):
    i = h.find(f'<section id="{sid}"')
    return h[i:h.find("</section>", i)] if i >= 0 else ""


def strip(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def test_zero_manshuu_day(gen):
    h = gen(data([danger("桐生", 1, 85), danger("桐生", 2, 70)], []), False, True)
    assert 'id="manshuu"' not in h and dangling(h) == []


def test_zero_danger_day_manshuu_only_ranking(gen):
    """危険艇0件 → ランキング・TOP5 は万舟だけのレース。行は残り、リンクだけ外れる"""
    dat = data([], [manshuu("戸田", r, 90 - r) for r in range(1, 13)])
    old = gen(dat, True, False)
    new = gen(dat, False, True)
    assert dangling(new) == []
    for sid in ("ai-ranking", "top5", "pickup"):
        assert strip(section(old, sid)) == strip(section(new, sid)), sid
    assert section(new, "ai-ranking").count('data-nolink="1"') == section(old, "ai-ranking").count('href="#race-')


def test_danger_plus_manshuu_index_icon_not_link_and_b2_title(gen):
    dat = data([danger("住之江", 5, 88)], [manshuu("住之江", 5, 82), manshuu("住之江", 9, 70)])
    h = gen(dat, False, True)
    idx = section(h, "race-index")
    assert "住之江9R" not in idx                       # 万舟だけの行は出さない
    assert '<span class="idx-icon-link" title="万舟">💰</span>' in idx
    assert 'title="AI万舟警報"' not in h               # B2：ツールチップは「万舟」
    assert 'href="#race-' in idx                        # 実在する危険艇カードへのリンクは残る


def test_flags_restore_previous_output(gen, tmp_path, monkeypatch):
    """切り替えを元に戻すと、万舟の一覧・リンク・ツールチップが元どおり出る"""
    dat = data([danger("住之江", 5, 88)], [manshuu("住之江", 5, 82), manshuu("住之江", 9, 70)])
    h = gen(dat, True, False)
    assert 'id="manshuu"' in h and 'title="AI万舟警報"' in h and "住之江9R" in section(h, "race-index")


def test_csv_and_markdown(tmp_path, monkeypatch):
    dat = data([danger("桐生", 1, 85)], [manshuu("桐生", 3, 81)])
    monkeypatch.setattr(n, "NEWSPAPER_SHOW_MANSHUU_LIST", False)
    paths = n.generate_csvs(dat, str(tmp_path) + "/")
    assert not any(p.endswith("manshuu.csv") for p in paths)
    md = tmp_path / "n.md"
    n.generate_markdown(dat, str(md))
    t = md.read_text(encoding="utf-8")
    assert 'id="manshuu"' not in t and "万舟Sランク" not in t


def test_x_post_without_manshuu():
    import x_post_text as xp
    s = xp.newspaper_post(12, None, "桐生", "1", 95)
    assert "万舟" not in s
    assert "万舟" in xp.newspaper_post(12, 30, "桐生", "1", 95)
