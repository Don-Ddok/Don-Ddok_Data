# -*- coding: utf-8 -*-
"""저축성 예금(거치식·적립식) × 요구불 관계 — 민영 요구불 사양으로 같은 계산 (탐색적, 집계만)

beta_sum_test.est 그대로(법인 FE + 달력월 더미, 보정 충격 ÷10, 법인·월 이중 군집, h마다 윈저라이즈).
계정별로 한 번이라도 잔액 > 0인 법인만. 전체 업종과 팀원 분석의 세 업종(1차 금속, 기타 기계, 도매·상품중개).
보유 여부 LP(1[잔액>0] 변화)도 같은 식으로 (윈저라이즈 없음과 결과가 같도록 값 범위 −1/0/1 그대로).
출력: 분석결과\\통합분석\\savings_lp.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\test\code")
import beta_sum_test as B

DATA = r"C:\test\data\processed\df_ready.csv"
TRIG = r"C:\test\분석결과\matching\trigger_stages.csv"
OUT = r"C:\test\분석결과\통합분석\savings_lp.csv"
IND = {"전체": None, "1차 금속": "1차 금속 제조업", "기타 기계": "기타 기계 및 장비 제조업", "도매·상품중개": "도매 및 상품 중개업"}
ACC = {"요구불": "요구불예금잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액"}
HS = [3, 6, 12]


def main():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "업종_중분류", "exposed", "exp_yoy"] + list(ACC.values()))
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d["지역"] = d["사업장_시도"]
    t = pd.read_csv(TRIG)[["지역", "연월", "보정YoY"]].rename(columns={"연월": "기준년월"})
    d = d.merge(t, on=["지역", "기준년월"]).sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
    d["midx"] = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
    rows = []
    for ik, iv in IND.items():
        di = d if iv is None else d[d["업종_중분류"] == iv]
        for ak, ac in ACC.items():
            if ik != "전체" and ak == "요구불":
                pass
            ever = di.groupby("법인ID")[ac].transform("max") > 0
            x = di[ever].copy()
            for kind in ["로그", "보유"]:
                x["ln_y"] = np.log(x[ac].clip(lower=0) + 1) if kind == "로그" else (x[ac] > 0).astype(float)
                if kind == "보유":
                    B.WINSOR = (0.0, 1.0)
                for h in HS:
                    try:
                        r = B.est(x, h, "보정YoY")
                    except Exception as e:          # 표본이 너무 작은 칸
                        print("skip", ik, ak, kind, h, e); continue
                    rows.append({"업종": ik, "계정": ak, "종속": kind, "h": h, "노출법인": int(x.loc[x.exposed == 1, "법인ID"].nunique()),
                                 "비노출법인": int(x.loc[x.exposed == 0, "법인ID"].nunique()),
                                 "노출_효과": round(-r["b13"] * (100 if kind == "로그" else 1), 3),
                                 "차이_효과": round(-r["b3"] * (100 if kind == "로그" else 1), 3), "p_차이": round(r["p3"], 4),
                                 "p_노출": round(r["p13"], 4), "G월": r["G월"]})
                B.WINSOR = (0.01, 0.99)
            print(ik, ak, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
