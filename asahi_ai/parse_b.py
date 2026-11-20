"""B ファイル（番組表）の解析。固定長のバイト位置で読み、黙って捨てない。"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from . import codes
from .layout import B_COLS, B_RACER_LEN, VENUE_CODES, ZEN2HAN

RE_RACER = re.compile(rb"^[1-6] \d{4}")
RE_RACE = re.compile(r"^[\s　]*([０-９\d]{1,2})Ｒ\s*(.*?)\s*Ｈ([０-９\d]{3,4})ｍ\s*電話投票締切予定([０-９\d]{1,2})：([０-９\d]{2})")
RE_DATE = re.compile(r"([０-９\d]{4})年[\s　]*([０-９\d]{1,2})月[\s　]*([０-９\d]{1,2})日")
RE_DAY = re.compile(r"第[\s　]*([０-９\d]{1,2})[\s　]*日|(初)[\s　]*日|(最[\s　]*終)[\s　]*日")


PLACEHOLDER_WORDS = ("いましばらくお待ちください", "全レース終了後に登録")


@dataclass
class BParse:
    races: list[dict] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)


def _num(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def _int(s: str):
    s = s.strip()
    return int(s) if s.isdigit() else None


def parse_b(txt: bytes, date: dt.date) -> BParse:
    out = BParse()
    venue = None
    race = None
    meet_day = None
    meet_final = False
    header_date_ok = None
    header_text = ""
    block_head: list[str] = []
    md_ok = None
    for lineno, raw in enumerate(txt.split(b"\r\n"), 1):
        m = re.fullmatch(rb"(\d{2})BBGN", raw)
        if m:
            venue = m.group(1).decode()
            race, meet_day, meet_final, header_date_ok, header_text, md_ok = None, None, False, None, "", None
            block_head = []
            if venue not in VENUE_CODES or venue in out.venues:
                out.issues.append({"level": "stop", "check": "b_venue_code", "venue": venue, "line": lineno})
            out.venues.append(venue)
            continue
        if re.fullmatch(rb"\d{2}BEND", raw):
            if venue is not None and header_date_ok is None and md_ok is True:
                out.issues.append({"level": "warn", "check": "b_header_year_missing", "venue": venue,
                                   "detail": header_text})
            elif venue is not None and header_date_ok is None and any(w in x for x in block_head for w in PLACEHOLDER_WORDS):
                # 「この場のデータ更新は、いましばらくお待ちください。」などの仮の場（データが無い）
                out.issues.append({"level": "warn", "check": "b_venue_placeholder", "venue": venue,
                                   "detail": block_head})
            elif venue is not None and header_date_ok is not True:
                out.issues.append({"level": "stop", "check": "b_header_date", "venue": venue,
                                   "detail": {"header": header_text, "block_head": block_head}})
            venue = None
            continue
        if venue is None:
            continue
        if RE_RACER.match(raw):
            if len(raw) != B_RACER_LEN:
                out.issues.append({"level": "stop", "check": "b_racer_len", "venue": venue,
                                   "race_no": race, "line": lineno, "detail": len(raw)})
                continue
            if race is None:
                out.issues.append({"level": "stop", "check": "b_racer_before_race", "venue": venue, "line": lineno})
                continue
            try:
                rec = {n: raw[a:b].decode("cp932").strip(" 　") for a, b, n in B_COLS}
            except UnicodeDecodeError:
                out.issues.append({"level": "stop", "check": "b_decode", "venue": venue, "line": lineno})
                continue
            entry = {
                "race_id": race["race_id"],
                "lane": int(rec["lane"]),
                "racer_no": rec["racer_no"],
                "racer_name": rec["racer_name"].replace("　", ""),
                "age": _int(rec["age"]),
                "branch": rec["branch"],
                "weight": _int(rec["weight"]),
                "class": rec["class"],
                "nat_win": _num(rec["nat_win"]),
                "nat_2r": _num(rec["nat_2r"]),
                "loc_win": _num(rec["loc_win"]),
                "loc_2r": _num(rec["loc_2r"]),
                "motor_no": _int(rec["motor_no"]),
                "motor_2r": _num(rec["motor_2r"]),
                "boat_no": _int(rec["boat_no"]),
                "boat_2r": _num(rec["boat_2r"]),
                "series_raw": rec["series_raw"],
                "series_orders": "".join(codes.parse_series(rec["series_raw"])),
                "hayami": rec["hayami"],
            }
            bad = [k for k in ("age", "weight", "nat_win", "nat_2r", "loc_win", "loc_2r",
                               "motor_no", "motor_2r", "boat_no", "boat_2r") if entry[k] is None]
            if entry["class"] not in ("A1", "A2", "B1", "B2"):
                bad.append("class")
            if bad:
                out.issues.append({"level": "quarantined", "check": "b_value", "venue": venue,
                                   "race_id": race["race_id"], "lane": entry["lane"], "detail": bad})
            out.entries.append(entry)
            continue
        s = raw.decode("cp932", errors="replace")
        if s.strip() and len(block_head) < 8:
            block_head.append(s.rstrip()[:80])
        mr = RE_RACE.match(s)
        if mr:
            no = int(mr.group(1).translate(ZEN2HAN))
            race = {
                "race_id": f"{date:%Y%m%d}{venue}{no:02d}",
                "date": date.isoformat(),
                "venue": venue,
                "race_no": no,
                "race_name": mr.group(2).replace("　", " ").strip(),
                "distance_m": int(mr.group(3).translate(ZEN2HAN)),
                "deadline_time": f"{int(mr.group(4).translate(ZEN2HAN)):02d}:{mr.group(5).translate(ZEN2HAN)}",
                "meet_day": meet_day,
                "meet_final_day": meet_final,
            }
            out.races.append(race)
            continue
        if re.match(r"^[\s　]*[０-９\d]{1,2}Ｒ", s):  # レースの見出しの形なのに読めない
            out.issues.append({"level": "stop", "check": "b_race_header", "venue": venue, "line": lineno,
                               "detail": s.strip()[:80]})
            race = None
            continue
        if race is None:
            mmd = re.search(r"^\S.*?[\s　]([０-９\d]{1,2})月[\s　]*([０-９\d]{1,2})日", s)
            if mmd and not RE_DATE.search(s) and md_ok is None:  # 場名の行「ボートレース大　村   　１月　１日」
                md_ok = (int(mmd.group(1).translate(ZEN2HAN)), int(mmd.group(2).translate(ZEN2HAN))) == (date.month, date.day)
                header_text = s.strip()[:80]
            md = RE_DATE.search(s)
            if md:  # 「第　４日          ２０２３年　１月　１日」の行
                y, mo, d = (int(md.group(i).translate(ZEN2HAN)) for i in (1, 2, 3))
                header_date_ok = (y, mo, d) == (date.year, date.month, date.day)
                header_text = s.strip()[:80]
                mday = RE_DAY.search(s)
                if mday:
                    meet_day = (int(mday.group(1).translate(ZEN2HAN)) if mday.group(1)
                                else 1 if mday.group(2) else None)
                    meet_final = bool(mday.group(3))
            continue
    if venue is not None:
        out.issues.append({"level": "stop", "check": "b_unterminated_block", "venue": venue})
    return out
