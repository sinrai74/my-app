"""K ファイル（競走成績）の解析。選手行・気象・決まり手・払戻。黙って捨てない。"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from . import codes
from .layout import BET_TYPES, K_COLS, K_RACER_LEN, VENUE_CODES

RE_RACER = re.compile(rb"^  [0-9A-Z ]{2}  [1-6] \d{4}")
RE_RACE = re.compile(r"^\s+(\d{1,2})R\s+(.*?)\s+H(\d{3,4})m\s+(.*)$")
RE_COND = re.compile(r"^(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm\s*$")
RE_DATE = re.compile(r"(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})")
RE_BET = re.compile(r"^\s{8}(単勝|複勝|２連単|２連複|拡連複|３連単|３連複)\s+(.*)$")
RE_BET_CONT = re.compile(r"^\s{17}(\d-\d.*)$")
RE_PAY_ITEM = re.compile(r"(特払い|返還|\d(?:-\d){0,2})\s+(\d+)(?:\s+人気\s+(\d+))?|(不成立)")
RE_CANCEL = re.compile(r"^\s+(\d{1,2})R\s+中\s*止\s*$")
RE_SUMMARY = re.compile(r"^\s+(\d{1,2})R\s+(\d-\d-\d|\S+)\s+(\d+)\s+")
RE_SUMMARY_HEAD = re.compile(r"^\s+\[払戻金\]")


PLACEHOLDER_WORDS = ("いましばらくお待ちください", "全レース終了後に登録")


@dataclass
class KParse:
    races: list[dict] = field(default_factory=list)      # K にあったレース（race_id・気象・決まり手）
    cancelled: dict = field(default_factory=dict)        # race_id → K の原文（「NR  中 止」）
    results: list[dict] = field(default_factory=list)
    payouts: list[dict] = field(default_factory=list)
    summary_trifecta: dict = field(default_factory=dict)  # race_id → (組合せ, 金額)。要約の表から
    issues: list[dict] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)


def _int(s: str):
    s = s.strip()
    return int(s) if s.isdigit() else None


def parse_k(txt: bytes, date: dt.date) -> KParse:
    out = KParse()
    venue = race = None
    bet = None
    header_date_ok = None
    header_text = ""
    block_head: list[str] = []
    in_summary = False
    md_ok = None
    seen_race = False
    for lineno, raw in enumerate(txt.split(b"\r\n"), 1):
        m = re.fullmatch(rb"(\d{2})KBGN", raw)
        if m:
            venue = m.group(1).decode()
            race, bet, header_date_ok, in_summary, header_text, md_ok, seen_race = None, None, None, False, "", None, False
            block_head = []
            if venue not in VENUE_CODES or venue in out.venues:
                out.issues.append({"level": "stop", "check": "k_venue_code", "venue": venue, "line": lineno})
            out.venues.append(venue)
            continue
        if re.fullmatch(rb"\d{2}KEND", raw):
            if venue is not None and header_date_ok is None and md_ok is True:
                # 年つきの日付の行が無い（場の全レース中止の日など）。場名の行の月日が一致すれば記録だけ
                out.issues.append({"level": "warn", "check": "k_header_year_missing", "venue": venue,
                                   "detail": header_text})
            elif venue is not None and header_date_ok is None and any(w in x for x in block_head for w in PLACEHOLDER_WORDS):
                # 「この場のデータ更新は、いましばらくお待ちください。」などの仮の場（データが無い）
                out.issues.append({"level": "warn", "check": "k_venue_placeholder", "venue": venue,
                                   "detail": block_head})
            elif venue is not None and header_date_ok is not True:
                out.issues.append({"level": "stop", "check": "k_header_date", "venue": venue,
                                   "detail": {"header": header_text, "block_head": block_head}})
            venue = None
            continue
        if venue is None:
            continue
        if RE_RACER.match(raw):
            bet = None
            if len(raw) != K_RACER_LEN:
                out.issues.append({"level": "stop", "check": "k_racer_len", "venue": venue,
                                   "line": lineno, "detail": len(raw)})
                continue
            if race is None:
                out.issues.append({"level": "stop", "check": "k_racer_before_race", "venue": venue, "line": lineno})
                continue
            try:
                rec = {n: raw[a:b].decode("cp932").strip(" 　") for a, b, n in K_COLS}
            except UnicodeDecodeError:
                out.issues.append({"level": "stop", "check": "k_decode", "venue": venue, "line": lineno})
                continue
            cls = codes.classify_result(rec["result_code_raw"])
            st_flag, st_value, st_ok = codes.parse_st(rec["st_raw"])
            ex, ex_ok = codes.parse_exhibition(rec["exhibition_raw"])
            rt, rt_ok = codes.parse_race_time(rec["race_time_raw"])
            course = int(rec["course_raw"]) if re.fullmatch(r"[1-6]", rec["course_raw"]) else None
            row = {
                "race_id": race["race_id"],
                "lane": int(rec["lane"]),
                "racer_no": rec["racer_no"],
                "racer_name": rec["racer_name"].replace("　", ""),
                "result_code_raw": rec["result_code_raw"],
                **cls,
                "course": course,
                "st_raw": rec["st_raw"],
                "st_flag": st_flag,
                "st_value": st_value,
                "exhibition_time": ex,
                "race_time_sec": rt,
                "motor_no": _int(rec["motor_no"]),
                "boat_no": _int(rec["boat_no"]),
            }
            bad = []
            if cls["result_class"] == "unknown":
                out.issues.append({"level": "quarantined", "check": "k_unknown_code", "venue": venue,
                                   "race_id": race["race_id"], "lane": row["lane"], "detail": rec["result_code_raw"]})
            if cls["result_class"] != "absent":
                if not st_ok:
                    bad.append("st")
                # 進入コースが無いのは、出走しなかった艇（出遅れで「L .」の艇など）。完走した艇で無ければ異常
                if course is None and (cls["result_class"] == "finished" or st_value is not None):
                    bad.append("course")
            if not ex_ok:
                bad.append("exhibition")
            if not rt_ok:
                bad.append("race_time")
            if row["motor_no"] is None or row["boat_no"] is None:
                bad.append("motor_boat")
            if bad:
                out.issues.append({"level": "quarantined", "check": "k_value", "venue": venue,
                                   "race_id": race["race_id"], "lane": row["lane"],
                                   "detail": {"fields": bad, "raw": raw.decode("cp932", errors="replace")}})
            out.results.append(row)
            continue
        s = raw.decode("cp932", errors="replace").replace("　", " ")
        if not s.strip():
            continue
        if len(block_head) < 8:
            block_head.append(s.rstrip()[:80])
        if RE_SUMMARY_HEAD.match(s):
            in_summary = True
            continue
        mr = RE_RACE.match(s)
        if mr:
            in_summary, bet, seen_race = False, None, True
            no = int(mr.group(1))
            cond = RE_COND.match(mr.group(4).strip())
            race = {
                "race_id": f"{date:%Y%m%d}{venue}{no:02d}",
                "venue": venue,
                "race_no": no,
                "race_name_k": mr.group(2).strip(),
                "distance_m": int(mr.group(3)),
                "weather": cond.group(1) if cond else None,
                "wind_dir": cond.group(2) if cond else None,
                "wind_speed_m": int(cond.group(3)) if cond else None,
                "wave_cm": int(cond.group(4)) if cond else None,
                "winning_technique": None,
                "race_void": False,
            }
            if not cond:
                out.issues.append({"level": "warn", "check": "k_conditions", "venue": venue,
                                   "race_id": race["race_id"], "detail": mr.group(4).strip()})
            out.races.append(race)
            continue
        if in_summary and not seen_race and re.match(r"^\s{9,}\d{1,2}R\s", s):
            # 払戻の要約の表（最初のレースの見出しより前）。中止・不成立の行もここに来る
            ms = RE_SUMMARY.match(s)
            mcan = RE_CANCEL.match(s)
            if mcan:  # 要約の表の「1R  中 止」：中止と明記されたレース
                out.cancelled[f"{date:%Y%m%d}{venue}{int(mcan.group(1)):02d}"] = s.strip()
            elif ms:
                rid = f"{date:%Y%m%d}{venue}{int(ms.group(1)):02d}"
                out.summary_trifecta[rid] = (ms.group(2), int(ms.group(3)))
            continue
        mcan = RE_CANCEL.match(s)
        if mcan:  # 「1R  中 止」：K に中止と明記されたレース
            rid = f"{date:%Y%m%d}{venue}{int(mcan.group(1)):02d}"
            out.cancelled[rid] = s.strip()
            race, bet = None, None
            continue
        if re.match(r"^\s+\d{1,2}R\s", s):  # レースの見出しの形なのに読めない → 列の定義が変わった恐れ
            out.issues.append({"level": "stop", "check": "k_race_header", "venue": venue, "line": lineno,
                               "detail": s.strip()[:80]})
            race, bet = None, None
            continue
        if race is None:
            mmd = re.search(r"［成績］\s+(\d{1,2})/\s*(\d{1,2})", s)
            if mmd:  # 場名の行「鳴 門［成績］      9/ 1」
                md_ok = (int(mmd.group(1)), int(mmd.group(2))) == (date.month, date.day)
                header_text = s.strip()[:80]
            md = RE_DATE.search(s)
            if md:
                header_date_ok = tuple(int(md.group(i)) for i in (1, 2, 3)) == (date.year, date.month, date.day)
                header_text = s.strip()[:80]
            continue
        if s.strip() == "レース不成立":
            race["race_void"] = True
            bet = None
            continue
        if "ﾚｰｽﾀｲﾑ" in s:
            tail = s.split("ﾚｰｽﾀｲﾑ", 1)[1].strip()
            race["winning_technique"] = tail or None
            continue
        if set(s.strip()) == {"-"}:
            continue
        mb = RE_BET.match(s)
        mc = RE_BET_CONT.match(s) if not mb else None
        if mb or (mc and bet):
            if mb:
                bet = BET_TYPES[mb.group(1)]
                body = mb.group(2)
            else:
                body = mc.group(1)
            items = RE_PAY_ITEM.findall(body)
            rest = RE_PAY_ITEM.sub("", body).strip()
            if not body.strip():  # 「複勝」だけの行：その券種の払戻が無い
                out.payouts.append({"race_id": race["race_id"], "bet_type": bet, "combination": "",
                                    "payout_yen": None, "popularity": None, "payout_kind": "none"})
                continue
            if not items or rest:
                out.issues.append({"level": "quarantined", "check": "k_payout_line", "venue": venue,
                                   "race_id": race["race_id"], "detail": s.strip()})
            for combo, amount, pop, void in items:
                if void:  # 「拡連複   不成立」：その券種は不成立（金額なし）
                    out.payouts.append({"race_id": race["race_id"], "bet_type": bet, "combination": "",
                                        "payout_yen": None, "popularity": None, "payout_kind": "不成立"})
                    continue
                kind = "hit" if combo[0].isdigit() else combo
                out.payouts.append({
                    "race_id": race["race_id"], "bet_type": bet,
                    "combination": combo if kind == "hit" else "",
                    "payout_yen": int(amount), "popularity": int(pop) if pop else None,
                    "payout_kind": kind,
                })
            continue
        out.issues.append({"level": "warn", "check": "k_unrecognized_line", "venue": venue,
                           "race_id": race["race_id"], "detail": s.strip()[:80]})
    if venue is not None:
        out.issues.append({"level": "stop", "check": "k_unterminated_block", "venue": venue})
    return out
