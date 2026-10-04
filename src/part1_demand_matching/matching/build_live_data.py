# -*- coding: utf-8 -*-
"""추천 화면용 실제 데이터 (대구·경북 × 36개월 전체, 집계만 — 법인 ID·개별 잔액 없음)

입력: 분석결과\\matching\\recommend_firm_month.csv, persona_firm_month.csv, trigger_stages.csv,
      im뱅크상품\\기업상품_매핑.csv
출력: 분석결과\\matching\\demo\\live_data.json, 분석결과\\matching\\demo\\index.html (템플릿에 데이터 삽입)
실행: py -3.11 build_live_data.py   (매칭을 다시 돌린 뒤 이 스크립트만 다시 실행하면 화면이 갱신됨)
"""
import json
import os
from datetime import date

import pandas as pd
import yaml

BASE = r"C:\test\분석결과\matching"
OUT_DIR = os.path.join(BASE, "demo")
TEMPLATE = os.path.join(OUT_DIR, "index_template.html")
PRODUCTS = r"C:\test\im뱅크상품\기업상품_매핑.csv"
HOME_URL = "https://www.imbank.co.kr/dgb_ebz_main.jsp"   # iM뱅크 홈페이지 메인 (상품몰 첫 화면은 메뉴 정보 없이 열면 본문이 비어서 쓰지 않음)
DEFAULT_REGION = "경북"   # 첫 화면 기본 지역 (기본 월은 가장 최근 월)
MIN_INDUSTRY = 5          # 이보다 적은 업종은 '기타 업종'으로 합침
MIN_SUBINDUSTRY = 5       # 제조업 중분류도 같은 기준으로 '그 외 제조업'
ADJUST = "한도·조건 조정"  # 이미 보유해 조건 조정으로 바뀐 제안방식
RECS = [1, 2, 3]
BETA_CSV = os.path.join(r"C:\test\분석결과", "민영_β합검정.csv")   # 인사이트: 요구불 반응 (사전등록 재검정, 주 사양)
RULES = os.path.join(r"C:\test\code\matching", "match_rules.yaml")
INSIGHT_JSON = os.path.join(r"C:\test\분석결과\insight", "insight_data.json")   # code\insight_map.py 출력 (인사이트 관계 지도)
BETA_NOTE = "수출 YoY 10%p 하락 시 외환노출 법인 요구불 누적 6개월 약 3.9%, 12개월 약 5.0% 감소 (β₁+β₃ 재검정, Holm 보정 후 h=2~12 유의)"
GUIDE_TYPES = ("수출금융", "수입금융")   # 가입 상품이 아닌 외환 업무 안내 페이지


def persona_counts(g):
    return {p[:1]: int(n) for p, n in g["페르소나"].value_counts().items()}


def industries(exp):
    """업종 묶음: 5곳 미만 업종 → 기타 업종, 제조업은 중분류(5곳 미만 → 그 외 제조업)."""
    ind_n = exp["업종_대분류"].value_counts()
    big = set(ind_n[ind_n >= MIN_INDUSTRY].index)
    exp = exp.assign(업종=exp["업종_대분류"].where(exp["업종_대분류"].isin(big), "기타 업종"))
    out = []
    for name, g in exp.groupby("업종"):
        item = {"업종": name, "법인수": int(len(g)), "페르소나": persona_counts(g)}
        if name == "기타 업종":
            item["포함업종"] = sorted(g["업종_대분류"].unique().tolist())
        if name == "제조업":
            sub_n = g["업종_중분류"].value_counts()
            sub_big = set(sub_n[sub_n >= MIN_SUBINDUSTRY].index)
            gg = g.assign(중분류=g["업종_중분류"].where(g["업종_중분류"].isin(sub_big), "그 외 제조업"))
            item["중분류"] = [{"업종_중분류": s, "법인수": int(len(h)), "페르소나": persona_counts(h)}
                           for s, h in sorted(gg.groupby("중분류"),
                                              key=lambda kv: (kv[0] == "그 외 제조업", -len(kv[1])))]
        out.append(item)
    out.sort(key=lambda i: (i["업종"] == "기타 업종", -i["법인수"]))
    return out


def cell(g):
    """페르소나 × 세그먼트 칸: 가장 많은 추천 조합을 대표로, 제안방식은 보유 여부와 무관한 기본값."""
    names = [f"추천{i}_상품명" for i in RECS]
    combos = g.groupby(names).size()
    top = combos.idxmax()
    rows = g[(g[names] == list(top)).all(axis=1)]
    row = rows.iloc[0]
    recs = []
    for i in RECS:
        if not row[f"추천{i}_상품명"]:
            continue
        m = rows[f"추천{i}_제안방식"]
        base = m[m != ADJUST]
        recs.append({"상품명": row[f"추천{i}_상품명"], "추가요건": row[f"추천{i}_추가요건"],
                     "이용경로": row[f"추천{i}_이용경로"], "법인추천가능": row[f"추천{i}_법인추천가능"],
                     "제안방식": base.mode().iloc[0] if len(base) else ADJUST,
                     "한도조건조정_비율": round(float((m == ADJUST).mean()), 3)})
    return {"법인수": int(len(g)), "대표조합_법인수": int(len(rows)), "추천": recs,
            "제안메시지": row["제안메시지"], "근거_타이밍": row["근거_타이밍"]}


def insights():
    """인사이트 관계 지도 데이터: code\insight_map.py가 만든 집계를 그대로 싣는다 (먼저 실행 필요)."""
    with open(INSIGHT_JSON, encoding="utf-8") as f:
        return json.load(f)


def main():
    rec = pd.read_csv(os.path.join(BASE, "recommend_firm_month.csv"), dtype=str, low_memory=False).fillna("")
    per = pd.read_csv(os.path.join(BASE, "persona_firm_month.csv"), dtype=str,
                      usecols=["법인ID", "연월", "업종_중분류"]).fillna("")
    trig = pd.read_csv(os.path.join(BASE, "trigger_stages.csv"))
    prod = pd.read_csv(PRODUCTS, dtype=str).fillna("")
    x = rec.merge(per, on=["법인ID", "연월"], how="left")
    x["업종_중분류"] = x["업종_중분류"].fillna("")

    used = set()
    regions = {}
    for (reg, ym), g in x.groupby(["지역", "연월"]):
        t = trig[(trig["지역"] == reg) & (trig["연월"] == int(ym))].iloc[0]
        exp = g[g["세그먼트"] != "비노출"]
        none = g[g["세그먼트"] == "비노출"]
        cells = []
        for (p, seg), h in exp.groupby(["페르소나", "세그먼트"]):
            c = cell(h)
            c.update({"페르소나": p, "세그먼트": seg})
            cells.append(c)
            used.update(r["상품명"] for r in c["추천"])
        gen = cell(none) if len(none) else None
        if gen:
            used.update(r["상품명"] for r in gen["추천"])
        regions.setdefault(reg, {})[ym] = {
            "단계": t["단계"], "k": None if pd.isna(t["k"]) else int(t["k"]),
            "T0": None if pd.isna(t["T0"]) else str(int(t["T0"])), "보정YoY": round(float(t["보정YoY"]), 1),
            "좌측절단": t["좌측절단"] == "Y",
            "요약": {"노출법인": int(len(exp)), "수출형": int((exp["세그먼트"] == "수출형").sum()),
                   "수입형": int((exp["세그먼트"] == "수입형").sum()), "비노출법인": int(len(none))},
            "업종": industries(exp) if len(exp) else [],
            "추천": cells,
            "비노출_일반": gen,
        }

    products = {}
    for n in sorted(used):
        r = prod[prod["상품명"] == n].iloc[0]
        clean = lambda v: "" if v in ("", "파싱실패") else " ".join(v.split())
        products[n] = {"모델유형": r["모델유형"], "대상": r["대상"], "금리기준일": r["금리기준일"],
                       "상세URL": r["상세URL"],   # 상품 원문 페이지 (사이트 상단 메뉴 없이 본문만 열림)
                       "안내페이지": r["모델유형"] in GUIDE_TYPES,
                       "목록": r["출처목록"].split("|")[0], "수집일": r["수집일"],
                       # 상세 창에 보여 줄 수집 원문 요약
                       "상세": {k: clean(r[c]) for k, c in [("금리", "금리"), ("한도", "한도"), ("기간", "기간"),
                                                          ("상환방식", "상환방식"), ("담보·보증", "담보·보증"),
                                                          ("가입대상", "가입대상원문"), ("상품설명", "상품설명원문")]}}

    months = sorted({ym for v in regions.values() for ym in v})
    data = {
        "meta": {"생성일": date.today().isoformat(), "기본지역": DEFAULT_REGION, "기본월": months[-1],
                 "월": months, "근거": BETA_NOTE, "홈페이지": HOME_URL,
                 "주의": "집계 데이터. 법인 ID·개별 잔액 없음. 제안메시지·기본순위·이용경로는 설계값(초안)"},
        "페르소나설명": {
            "A": {"이름": "A 결제형", "정의": "할인어음·무역금융 보유"},
            "B": {"이름": "B 차입운영형", "정의": "순수 운전자금대출 보유"},
            "C": {"이름": "C 설비투자형", "정의": "시설자금대출 보유"},
            "D": {"이름": "D 현금비축형", "정의": "거치식·적립식 예금 보유"},
            "E": {"이름": "E 요구불중심형", "정의": "위 계정 없음"}},
        "상품": products,
        "지역": regions,
        "인사이트": insights(),
    }
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    assert "법인ID" not in js
    with open(os.path.join(OUT_DIR, "live_data.json"), "w", encoding="utf-8") as f:
        f.write(js)
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read().replace("__DATA_JSON__", js.replace("</", "<\\/"))
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # 점검: 업종 합 = 노출 법인, 칸 합 = 노출 법인
    for reg, v in regions.items():
        for ym, m in v.items():
            s = m["요약"]["노출법인"]
            assert sum(i["법인수"] for i in m["업종"]) == s, (reg, ym)
            assert sum(c["법인수"] for c in m["추천"]) == s, (reg, ym)
    alt = sum(c["법인수"] - c["대표조합_법인수"] for v in regions.values() for m in v.values() for c in m["추천"])
    print(f"저장: live_data.json ({len(js) / 1024:.0f} KB), index.html | 지역 {list(regions)} × {len(months)}개월 "
          f"| 상품 {len(products)}개 | 대표조합과 다른 노출 법인-월 {alt}")
    for reg in regions:
        m = regions[reg][months[-1]]
        print(f"  {reg} {months[-1]}: {m['단계']} k={m['k']} 노출 {m['요약']['노출법인']} 비노출 {m['요약']['비노출법인']}")


if __name__ == "__main__":
    main()
