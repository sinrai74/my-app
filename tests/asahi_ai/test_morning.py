"""朝の番組表（P4）の変換と照合の試験。"""
from __future__ import annotations

import datetime as dt
import json
import unittest

from asahi_ai import morning

D = dt.date(2026, 11, 24)


def program_json(boats=6, cls=1, date="2026-11-24"):
    return json.dumps({"programs": [{
        "race_date": date, "race_stadium_number": 9, "race_number": 3, "race_closed_at": f"{date} 11:52:00",
        "race_distance": 1800,
        "boats": [{"racer_boat_number": i, "racer_number": 4000 + i, "racer_class_number": cls, "racer_age": 30,
                   "racer_weight": 52.4, "racer_national_top_1_percent": 6.1, "racer_national_top_2_percent": 40.5,
                   "racer_local_top_1_percent": 5.0, "racer_local_top_2_percent": 30.0,
                   "racer_assigned_motor_number": 10 + i, "racer_assigned_motor_top_2_percent": 33.3,
                   "racer_assigned_boat_number": 20 + i, "racer_assigned_boat_top_2_percent": 31.0}
                  for i in range(1, boats + 1)]}]}).encode("utf-8")


class TestMorning(unittest.TestCase):
    def test_to_tables(self):
        races, entries, issues = morning.to_tables(program_json(), D)
        self.assertEqual(issues, [])
        self.assertEqual(races.iloc[0]["race_id"], "202611240903")
        self.assertEqual(races.iloc[0]["deadline_time"], "11:52")
        e = entries.iloc[0]
        self.assertEqual((e["racer_no"], e["class"], e["motor_no"]), ("4001", "A1", 11))

    def test_class_map_and_not6(self):
        _, entries, issues = morning.to_tables(program_json(boats=5, cls=4), D)
        self.assertEqual(entries.iloc[0]["class"], "B2")
        self.assertIn(("morning_not6", "quarantined"), [(i["check"], i["level"]) for i in issues])

    def test_wrong_date_stops(self):
        _, _, issues = morning.to_tables(program_json(date="2026-11-23"), D)
        self.assertEqual(issues[0]["level"], "stop")

    def test_compare_with_b(self):
        _, entries, _ = morning.to_tables(program_json(), D)
        b = [{"race_id": "202611240903", "lane": i, "racer_no": f"{4000 + i:04d}", "nat_win": 6.1, "nat_2r": 40.5,
              "motor_no": 10 + i, "motor_2r": 33.3, "boat_no": 20 + i, "boat_2r": 31.0, "age": 30, "class": "A1"}
             for i in range(1, 7)]
        b[2]["motor_2r"] = 35.0
        c = morning.compare_with_b(entries, b)
        self.assertEqual((c["matched"], c["diff"], c["missing_in_morning"]), (6, {"motor_2r": 1}, 0))


if __name__ == "__main__":
    unittest.main()


class TestBFallback(unittest.TestCase):
    def test_b_to_tables_shape(self):
        import datetime as dt
        from types import SimpleNamespace
        from asahi_ai import daily
        d = dt.date(2026, 11, 24)
        e = {"race_id": "202611240101", "lane": 1, "racer_no": "4444", "age": 30, "weight": 52, "class": "A1",
             "nat_win": 7.1, "nat_2r": 50.0, "loc_win": 6.0, "loc_2r": 40.0, "motor_no": 11, "motor_2r": 33.3,
             "boat_no": 22, "boat_2r": 30.0, "series_raw": "x", "series_orders": "12", "hayami": ""}
        b = SimpleNamespace(races=[{"race_id": "202611240101", "venue": "01", "race_no": 1, "distance_m": 1800,
                                    "deadline_time": "10:45", "meet_day": 3}],
                            entries=[e], issues=[{"level": "warn", "check": "x"}])
        races, entries, issues = daily.b_to_tables(b, d)
        self.assertEqual(races.loc[0, "race_closed_at"], "2026-11-24 10:45:00")
        self.assertIsNone(races.loc[0, "meet_day"])            # 朝の番組表に無い項目は入れない
        self.assertEqual(entries.loc[0, "series_orders"], "")  # 同上
        self.assertEqual(issues, [])
