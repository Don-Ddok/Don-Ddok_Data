# -*- coding: utf-8 -*-
"""자체 작성 달력: 월별 영업일수 (2022-01 ~ 2025-12)
영업일수 = 그 달 평일(월~금) 수 - 평일에 낀 공휴일 수
근로자의 날(5/1)·12/31은 공휴일로 치지 않음. 대체·임시·선거일 포함.
공휴일 목록은 팀 지시 목록을 그대로 옮김 (보고서 달력과 다를 수 있음).
"""
import sys, calendar, datetime as dt
import pandas as pd

HOLIDAYS = {
    2022: ["01-31", "02-01", "02-02", "03-01", "03-09", "05-05", "06-01", "06-06",
           "08-15", "09-09", "09-12", "10-03", "10-10"],
    2023: ["01-23", "01-24", "03-01", "05-05", "05-29", "06-06", "08-15",
           "09-28", "09-29", "10-02", "10-03", "10-09", "12-25"],
    2024: ["01-01", "02-09", "02-12", "03-01", "04-10", "05-06", "05-15",
           "06-06", "08-15", "09-16", "09-17", "09-18", "10-01", "10-03", "10-09", "12-25"],
    2025: ["01-01", "01-27", "01-28", "01-29", "01-30", "03-03", "05-05", "05-06",
           "06-03", "06-06", "08-15", "10-03", "10-06", "10-07", "10-08", "10-09", "12-25"],
}
OUT = r"C:\test\data\ext_bizday.csv"

hol = [dt.date(y, int(md[:2]), int(md[3:])) for y, lst in HOLIDAYS.items() for md in lst]
assert len(hol) == len(set(hol)), "중복 날짜 있음"
weekend = [d for d in hol if d.weekday() >= 5]
if weekend:
    print("[중단] 주말인 공휴일:", [d.isoformat() for d in weekend]); sys.exit(1)
print(f"공휴일 {len(hol)}개 모두 평일 확인")

rows = []
for y in range(2022, 2026):
    for m in range(1, 13):
        nd = calendar.monthrange(y, m)[1]
        wd = sum(dt.date(y, m, d).weekday() < 5 for d in range(1, nd + 1))
        h = sum(1 for d in hol if d.year == y and d.month == m)
        rows.append(dict(ym=y * 100 + m, weekdays=wd, holidays=h, bizdays=wd - h))
B = pd.DataFrame(rows)
assert len(B) == 48
B[["ym", "bizdays"]].to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"저장: {OUT} ({len(B)}개월)")
print(B.pivot_table(index=B.ym // 100, columns=B.ym % 100, values="bizdays").astype(int).to_string())
