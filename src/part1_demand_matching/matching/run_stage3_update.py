# -*- coding: utf-8 -*-
"""3단계 수정 반영 실행기 (2026-09-28): 수정 전 지표 → 상품표 재생성 → 매칭 재실행 → 수정 후 지표·md 집계

순서
  1) 현재 recommend_firm_month.csv(수정 전 결과)로 비교 지표 계산 → stage3_before_metrics.json
     (이미 있으면 다시 계산하지 않음 — 재실행해도 '수정 전' 값이 덮어써지지 않게)
  2) im뱅크상품\\map_products.py 실행 (지역 필터 반영된 상품표 재생성)
  3) matcher.py 실행
  4) 수정 후 지표와 md용 집계 출력 + 집계 CSV 저장
실행: py -3.11 run_stage3_update.py
"""
import json
import os
import subprocess
import sys

import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\test\분석결과\matching"
REC = os.path.join(BASE, "recommend_firm_month.csv")
PER = os.path.join(BASE, "persona_firm_month.csv")
BEFORE = os.path.join(BASE, "stage3_before_metrics.json")
HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCT_DIR = r"C:\test\im뱅크상품"
RECS = [1, 2, 3]


def load_rec():
    d = pd.read_csv(REC, dtype=str, low_memory=False).fillna("")
    d["연월"] = d["연월"].astype(int)
    return d


def metrics(d: pd.DataFrame, rules: dict) -> dict:
    """비교 지표: 관찰 단계 운전자금 부착 행, 정점 보증정책연계 3칸 독점 행, 지역 불일치 행."""
    types = [f"추천{i}_모델유형" for i in RECS]
    names = [f"추천{i}_상품명" for i in RECS]
    cells = {k: set(v["types"]) for k, v in rules["stage_persona"]["관찰"].items()}
    ob = d[d["단계"] == "관찰"]
    attach = ob.apply(lambda r: "운전자금" in {r[c] for c in types if r[c]} - cells[r["페르소나"][:1]], axis=1).sum()
    pk = d[d["단계"] == "정점·유지"]
    mono = (pk[types] == "보증정책연계").all(axis=1).sum()

    def mism(r):
        for c in names:
            n = r[c]
            if r["지역"] == "대구" and any(x in n for x in ("경상북도", "경북", "포항", "구미")):
                return True
            if r["지역"] == "경북" and "대구" in n:
                return True
        return False
    region = d.apply(mism, axis=1).sum()
    return {"관찰_운전자금_부착_행": int(attach), "관찰_행": int(len(ob)),
            "정점_보증정책연계_3칸독점_행": int(mono), "정점_행": int(len(pk)),
            "지역불일치_행": int(region), "전체_행": int(len(d))}


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    old_rules = yaml.safe_load(open(os.path.join(HERE, "match_rules.yaml"), encoding="utf-8"))
    if not os.path.exists(BEFORE):
        before = metrics(load_rec(), old_rules)
        json.dump(before, open(BEFORE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    before = json.load(open(BEFORE, encoding="utf-8"))
    print("[수정 전]", before)

    subprocess.run([sys.executable, "map_products.py"], cwd=PRODUCT_DIR, check=True, env=env,
                   stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-W", "ignore", "matcher.py"], cwd=HERE, check=True, env=env)

    rules = yaml.safe_load(open(os.path.join(HERE, "match_rules.yaml"), encoding="utf-8"))
    d = load_rec()
    after = metrics(d, rules)
    cmp_file = os.path.join(BASE, "stage3_compare_metrics.json")
    prev = json.load(open(cmp_file, encoding="utf-8")).get("after") if os.path.exists(cmp_file) else None
    if prev:
        print("[직전 실행]", prev)
    print("[이번 실행]", after)
    json.dump({"before": before, "previous": prev, "after": after}, open(cmp_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 추천 속성 비율 (출력이 잘리지 않도록 앞쪽에 두고 CSV로도 저장)
    s = d[d["연월"] == 202512]

    def slots(x):
        out = []
        for i in RECS:
            z = x[x[f"추천{i}_상품명"] != ""][[f"추천{i}_법인추천가능", f"추천{i}_추가요건", f"추천{i}_제안방식"]]
            z.columns = ["rec", "req", "method"]
            out.append(z)
        return pd.concat(out)
    attr_rows = []
    print("\n[추천 속성 비율]")
    for nm, x in [("전체", d), ("노출 36개월", d[d["세그먼트"] != "비노출"]), ("2025-12", s),
                  ("2025-12 노출", s[s["세그먼트"] != "비노출"])]:
        z = slots(x)
        adj = (x[[f"추천{i}_제안방식" for i in RECS]] == "한도·조건 조정").any(axis=1).mean()
        row = {"범위": nm, "행": len(x), "슬롯": len(z), "확인필요(%)": round((z.rec == "확인필요").mean() * 100, 1),
               "추가요건(%)": round((z.req != "").mean() * 100, 1),
               "조정슬롯(%)": round((z.method == "한도·조건 조정").mean() * 100, 1), "조정행(%)": round(adj * 100, 1)}
        attr_rows.append(row)
        print(f"[{nm}] 행 {len(x):,} 슬롯 {len(z):,} | 확인필요 {row['확인필요(%)']}% | 추가요건 {row['추가요건(%)']}% | "
              f"조정 슬롯 {row['조정슬롯(%)']}% | 조정 행 {row['조정행(%)']}%")
    pd.DataFrame(attr_rows).to_csv(os.path.join(BASE, "recommend_summary_attributes.csv"), index=False,
                                   encoding="utf-8-sig")
    print("[근거_타이밍]", d["근거_타이밍"].value_counts().to_dict())

    # 추천 개수 분포 (단계별)
    n_rec = (d[[f"추천{i}_상품명" for i in RECS]] != "").sum(axis=1)
    print("\n[추천 개수 분포 — 단계별 행 수]")
    print(pd.crosstab(d["단계"], n_rec.rename("추천 개수"), margins=True, margins_name="합계").to_string())

    # 같은 모델유형 2개 이상 행: 칸 유형 1개(추천된 유형이 1종)인 경우만 남아야 함
    types_ = [f"추천{i}_모델유형" for i in RECS]
    kinds = d[types_].apply(lambda r: [t for t in r if t], axis=1)
    dup_rows = kinds.apply(lambda L: len(L) != len(set(L)))
    mixed = kinds.apply(lambda L: len(L) != len(set(L)) and len(set(L)) > 1)
    print(f"\n[같은 모델유형 2개 이상인 행] {int(dup_rows.sum()):,} / {len(d):,} | "
          f"그중 다른 유형과 섞인 행(0이어야 함): {int(mixed.sum())}")
    print(d[dup_rows].groupby(["단계", "페르소나"]).size().sort_values(ascending=False).head(10).to_string())

    # 데모 기준월 2025-07 경북 노출 법인: 페르소나별 추천 1·2·3순위
    demo7 = d[(d["연월"] == 202507) & (d["지역"] == "경북") & (d["세그먼트"] != "비노출")]
    cols3 = ["페르소나", "세그먼트"] + [f"추천{i}_상품명" for i in RECS] + ["제안메시지"]
    combo = demo7.groupby(cols3).size().rename("법인수").reset_index().sort_values(["페르소나", "법인수"],
                                                                                 ascending=[True, False])
    combo.to_csv(os.path.join(BASE, "recommend_demo_202507.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[데모 기준월 2025-07 경북 노출 법인 {len(demo7)}곳 — 페르소나별 추천 1·2·3순위]")
    print(combo.to_string(index=False))

    pd.set_option("display.width", 250)
    t = pd.crosstab(s["단계"], s["페르소나"].replace("", "(비노출)"), margins=True, margins_name="합계")
    t.to_csv(os.path.join(BASE, "recommend_summary_stage_persona_202512.csv"), encoding="utf-8-sig")
    print("\n[2025-12 단계 × 페르소나]\n", t.to_string())
    for label, x in [("노출", s[s["세그먼트"] != "비노출"]), ("비노출", s[s["세그먼트"] == "비노출"])]:
        top = x.groupby(["추천1_상품명", "추천1_모델유형"]).size().sort_values(ascending=False).head(10)
        top = top.rename("법인수").reset_index()
        top["비중(%)"] = (top["법인수"] / len(x) * 100).round(1)
        top.to_csv(os.path.join(BASE, f"recommend_summary_top10_202512_{label}.csv"), index=False, encoding="utf-8-sig")
        print(f"\n[2025-12 1순위 상위 10 — {label} {len(x)}곳]\n", top.to_string(index=False))

    # 데모 기준월 후보: 경북 적기 구간
    demo = d[(d["지역"] == "경북") & (d["단계"] == "적기") & d["연월"].between(202507, 202510)]
    print("\n[데모 후보월: 경북 적기 노출 법인 수]", demo.groupby("연월")["법인ID"].nunique().to_dict())
    dist = demo.groupby(["연월", "페르소나", "추천1_상품명"]).size().rename("법인수").reset_index()
    dist.to_csv(os.path.join(BASE, "recommend_summary_demo_candidates.csv"), index=False, encoding="utf-8-sig")
    print(dist.to_string(index=False))

    tr = d[d["단계"] == "적기"].groupby(["연월", "지역"])["법인ID"].nunique().unstack(fill_value=0)
    tr.to_csv(os.path.join(BASE, "recommend_summary_optimal_trend.csv"), encoding="utf-8-sig")

    print("\n[제안메시지 — 2025-07 경북 노출 A 결제형, 세그먼트별]")
    print(demo7[demo7["페르소나"].str.startswith("A")].groupby(["세그먼트", "제안메시지"]).size().to_string())


if __name__ == "__main__":
    main()
