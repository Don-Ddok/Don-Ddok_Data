# -*- coding: utf-8 -*-
"""마케팅 담당자 추천 화면 데모용 데이터 (집계만 — 법인 ID·개별 잔액 없음)

입력: 분석결과\\matching\\recommend_firm_month.csv, trigger_stages.csv, im뱅크상품\\기업상품_매핑.csv
출력: 분석결과\\matching\\demo\\demo_data.json
실행: py -3.11 build_demo_data.py
"""
import json
import os

import pandas as pd

BASE = r"C:\test\분석결과\matching"
OUT_DIR = os.path.join(BASE, "demo")
DEMO_YM, DEMO_REGION = "202507", "경북"
MIN_INDUSTRY = 5          # 이보다 적은 업종은 '기타 업종'으로 합침 (화면 단순화)
MIN_SUBINDUSTRY = 5       # 제조업 중분류도 같은 기준으로 '기타 제조업'
RECS = [1, 2, 3]
FIELDS = ["상품명", "모델유형", "법인추천가능", "추가요건", "제안방식", "이용경로"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rec = pd.read_csv(os.path.join(BASE, "recommend_firm_month.csv"), dtype=str, low_memory=False).fillna("")
    per = pd.read_csv(os.path.join(BASE, "persona_firm_month.csv"), dtype=str,
                      usecols=["법인ID", "연월", "업종_중분류"]).fillna("")
    trig = pd.read_csv(os.path.join(BASE, "trigger_stages.csv"))
    prod = pd.read_csv(r"C:\test\im뱅크상품\기업상품_매핑.csv", dtype=str).fillna("")

    x = rec[(rec["연월"] == DEMO_YM) & (rec["지역"] == DEMO_REGION)].merge(per, on=["법인ID", "연월"], how="left")
    exp = x[x["세그먼트"] != "비노출"].copy()
    none = x[x["세그먼트"] == "비노출"]

    # 업종 묶기
    ind_n = exp["업종_대분류"].value_counts()
    big = set(ind_n[ind_n >= MIN_INDUSTRY].index)
    exp["업종"] = exp["업종_대분류"].where(exp["업종_대분류"].isin(big), "기타 업종")
    sub_n = exp.loc[exp["업종_대분류"] == "제조업", "업종_중분류"].value_counts()
    sub_big = set(sub_n[sub_n >= MIN_SUBINDUSTRY].index)

    def persona_counts(g):
        return {p[:1]: int(n) for p, n in g["페르소나"].value_counts().items()}

    industries = []
    for name, g in exp.groupby("업종"):
        item = {"업종": name, "법인수": int(len(g)),
                "세그먼트": {k: int(v) for k, v in g["세그먼트"].value_counts().items()},
                "페르소나": persona_counts(g)}
        if name == "기타 업종":
            item["포함업종"] = sorted(g["업종_대분류"].unique().tolist())
        if name == "제조업":
            gg = g.assign(중분류=g["업종_중분류"].where(g["업종_중분류"].isin(sub_big), "그 외 제조업"))
            item["중분류"] = [{"업종_중분류": s, "법인수": int(len(h)), "페르소나": persona_counts(h)}
                           for s, h in sorted(gg.groupby("중분류"), key=lambda kv: -len(kv[1]))]
        industries.append(item)
    industries.sort(key=lambda i: (i["업종"] == "기타 업종", -i["법인수"]))

    # 페르소나 × 세그먼트 추천 (같은 칸은 같은 추천 — 조정 여부만 법인마다 다름)
    cells = []
    for (p, seg), g in exp.groupby(["페르소나", "세그먼트"]):
        top = g.groupby([f"추천{i}_상품명" for i in RECS]).size().idxmax()
        row = g[(g[[f"추천{i}_상품명" for i in RECS]] == list(top)).all(axis=1)].iloc[0]
        recs = []
        for i in RECS:
            if row[f"추천{i}_상품명"]:
                r = {f: row[f"추천{i}_{f}"] for f in FIELDS}
                adj = (g[f"추천{i}_제안방식"] == "한도·조건 조정").mean()
                r["제안방식"] = "선제 제안" if r["제안방식"] == "한도·조건 조정" else r["제안방식"]
                r["한도조건조정_비율"] = round(float(adj), 3)   # 이미 보유해 '한도·조건 조정'이 되는 법인 비율
                pinfo = prod[prod["상품명"] == r["상품명"]].iloc[0]
                r.update({"대상": pinfo["대상"], "상세URL": pinfo["상세URL"], "금리기준일": pinfo["금리기준일"]})
                recs.append(r)
        cells.append({"페르소나": p, "세그먼트": seg, "법인수": int(len(g)), "추천": recs,
                      "제안메시지": row["제안메시지"], "근거_타이밍": row["근거_타이밍"]})

    general = none.iloc[0]
    data = {
        "meta": {"기준월": "2025-07", "지역": DEMO_REGION, "단계": "적기", "k": 3,
                 "근거_타이밍": "확정 (β1+β3)",
                 "설명": "경북 수출 둔화 T0=2025-04, 적기(k=3~6) 첫 달. 수출 10%p 하락당 외환노출 법인 요구불 6개월 4.2% 감소",
                 "주의": "집계 데이터. 법인 ID·개별 잔액 없음. 제안메시지·기본순위·이용경로는 설계값(초안)"},
        "요약": {"노출법인": int(len(exp)), "수출형": int((exp["세그먼트"] == "수출형").sum()),
                "수입형": int((exp["세그먼트"] == "수입형").sum()), "비노출법인": int(len(none))},
        "페르소나설명": {
            "A": {"이름": "A 결제형", "정의": "할인어음·무역금융 보유"},
            "B": {"이름": "B 차입운영형", "정의": "순수 운전자금대출 보유"},
            "C": {"이름": "C 설비투자형", "정의": "시설자금대출 보유"},
            "D": {"이름": "D 현금비축형", "정의": "거치식·적립식 예금 보유"},
            "E": {"이름": "E 요구불중심형", "정의": "위 계정 없음"}},
        "업종": industries,
        "추천": cells,
        "비노출_일반": {"법인수": int(len(none)), "제안메시지": general["제안메시지"],
                     "추천": [general[f"추천{i}_상품명"] for i in RECS if general[f"추천{i}_상품명"]],
                     "근거_타이밍": general["근거_타이밍"]},
        "트리거": [{"지역": r["지역"], "연월": int(r["연월"]), "보정YoY": round(float(r["보정YoY"]), 1),
                  "단계": r["단계"]} for _, r in trig.iterrows()],
    }
    path = os.path.join(OUT_DIR, "demo_data.json")
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", path)
    print("요약:", data["요약"])
    for i in industries:
        print(" ", i["업종"], i["법인수"], i["페르소나"], [ (s["업종_중분류"], s["법인수"]) for s in i.get("중분류", [])])
    for c in cells:
        print(" ", c["페르소나"], c["세그먼트"], c["법인수"], [r["상품명"] for r in c["추천"]], c["제안메시지"])


if __name__ == "__main__":
    main()
