import csv
with open("hit_record.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
seen, unique = set(), []
for r in rows:
    k = (r["date"], r["venue_num"], r["race"], r["pred_combo"])
    if k not in seen:
        seen.add(k)
        unique.append(r)
with open("hit_record.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(unique)
print(f"{len(rows)}件 → {len(unique)}件に整理しました")