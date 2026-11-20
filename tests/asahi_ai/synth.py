"""試験用の合成ファイル（公式ファイルの形をまねた、架空の選手・値）。

公式の原本はリポジトリ（公開）に置かない。実ファイルでの試験は環境変数
ASAHI_TEST_RAW_DIR に b/k の LZH を置いたときだけ行う（test_real_files.py）。
"""
from __future__ import annotations

from asahi_ai.layout import B_COLS, B_RACER_LEN, K_COLS, K_RACER_LEN


def _fixed(cols, total, values: dict, align_right=()) -> bytes:
    buf = bytearray(b" " * total)
    for a, b, name in cols:
        v = values.get(name, "")
        raw = str(v).encode("cp932")
        width = b - a
        if len(raw) > width:
            raise ValueError(f"{name}={v!r} が {width} バイトを超える")
        raw = raw.rjust(width) if name in align_right else raw.ljust(width)
        buf[a:b] = raw
    assert len(buf) == total
    return bytes(buf)


def b_racer(lane, racer_no="4001", name="試験太郎", age="30", branch="東京", weight="52", cls="A1",
            nat_win="6.00", nat_2r="40.00", loc_win="5.50", loc_2r="35.00",
            motor_no="12", motor_2r="33.33", boat_no="45", boat_2r="30.00",
            series="1 2", hayami="") -> bytes:
    return _fixed(B_COLS, B_RACER_LEN, {
        "lane": lane, "racer_no": racer_no, "racer_name": name, "age": age, "branch": branch,
        "weight": weight, "class": cls, "nat_win": nat_win, "nat_2r": nat_2r, "loc_win": loc_win,
        "loc_2r": loc_2r, "motor_no": motor_no, "motor_2r": motor_2r, "boat_no": boat_no,
        "boat_2r": boat_2r, "series_raw": series, "hayami": hayami,
    }, align_right={"motor_no", "boat_no", "motor_2r", "boat_2r", "nat_2r", "loc_2r"})


def k_racer(code, lane, racer_no="4001", name="試　験　　太　郎", motor="12", boat="45",
            ex="6.80", course="1", st="0.10", rt="1.50.0") -> bytes:
    return _fixed(K_COLS, K_RACER_LEN, {
        "result_code_raw": code, "lane": lane, "racer_no": racer_no, "racer_name": name,
        "motor_no": motor, "boat_no": boat, "exhibition_raw": ex, "course_raw": course,
        "st_raw": st, "race_time_raw": rt,
    }, align_right={"motor_no", "boat_no", "exhibition_raw", "course_raw"})


def b_file(venue="09", venue_name="津", year=2023, month=1, day=1, day_label="第　４日",
           races=None) -> bytes:
    """races: [(race_no, [racer_line_bytes...]), ...]"""
    z = str.maketrans("0123456789:", "０１２３４５６７８９：")
    lines = [b"STARTB", f"{venue}BBGN".encode()]
    lines.append(f"ボートレース{venue_name}   　{month}月{day}日  試験杯  {day_label}".encode("cp932"))
    lines.append(b"")
    lines.append(f"   {day_label}          {str(year).translate(z)}年{str(month).translate(z).rjust(2, '　')}月"
                 f"{str(day).translate(z).rjust(2, '　')}日          ボートレース{venue_name}".encode("cp932"))
    lines.append(b"")
    for no, racers in races or []:
        lines.append(f"　{str(no).translate(z)}Ｒ  一般　　　　          Ｈ１８００ｍ  電話投票締切予定１５：２０ ".encode("cp932"))
        lines.append(b"-" * 79)
        lines.extend(racers)
        lines.append(b"")
    lines += [f"{venue}BEND".encode(), b"FINALB", b""]
    return b"\r\n".join(lines)


def k_file(venue="09", venue_name="津", year=2023, month=1, day=1, races=None, summary=None) -> bytes:
    """races: [(race_no, racer_lines, payout_lines(str), cond_str, technique), ...]"""
    lines = [b"STARTK", f"{venue}KBGN".encode()]
    lines.append(f"{venue_name}［成績］      {month}/{day:>2}      試験杯  第 4日".encode("cp932"))
    lines.append(b"")
    lines.append(f"   第 4日          {year}/{month:>2}/{day:>2}                             ボートレース{venue_name}".encode("cp932"))
    lines.append(b"")
    lines.append("   [払戻金]       ３連単           ３連複           ２連単         ２連複".encode("cp932"))
    for no, combo, amt in summary or []:
        lines.append(f"{no:>12}R  {combo}    {amt:>5}    1-2-3     100    1-2     100    1-2     100".encode("cp932"))
    lines += [b"", b""]
    for no, racers, pays, cond, tech in races or []:
        lines.append(f"{no:>4}R       一般　　　　                 H1800m  {cond}".encode("cp932"))
        lines.append(f"  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ {tech}".encode("cp932"))
        lines.append(b"-" * 79)
        lines.extend(racers)
        lines.append(b"")
        lines.extend(p.encode("cp932") for p in pays)
        lines += [b"", b""]
    lines += [f"{venue}KEND".encode(), b"FINALK", b""]
    return b"\r\n".join(lines)


def std_pays(order=(1, 2, 3)):
    a, b, c = order
    return [
        f"        単勝     {a}          110  ",
        f"        複勝     {a}          100  {b}          100  ",
        f"        ２連単   {a}-{b}        260  人気     1 ",
        f"        ２連複   {min(a, b)}-{max(a, b)}        180  人気     1 ",
        f"        拡連複   {min(a, b)}-{max(a, b)}        100  人気     1 ",
        f"                 {min(a, c)}-{max(a, c)}        170  人気     3 ",
        f"                 {min(b, c)}-{max(b, c)}        240  人気     7 ",
        f"        ３連単   {a}-{b}-{c}      600  人気     1 ",
        f"        ３連複   {'-'.join(map(str, sorted(order)))}      250  人気     1 ",
    ]


def std_k_racers(order=(1, 2, 3, 4, 5, 6), codes=None, sts=None):
    codes = codes or {}
    sts = sts or {}
    out = []
    for pos, lane in enumerate(order, 1):
        code = codes.get(lane, f"0{pos}")
        out.append(k_racer(code, lane, racer_no=f"40{lane:02d}", course=str(lane),
                           st=sts.get(lane, "0.10"), rt="1.50.0" if pos <= 3 else ".  ."))
    return out


def std_b_racers():
    return [b_racer(l, racer_no=f"40{l:02d}") for l in range(1, 7)]
