# -*- coding: utf-8 -*-
"""인사이트 관계 지도용 집계 (집계만 — 법인 ID·개별 잔액 없음)

사전등록: 분석결과\\민영_법인클러스터_사전등록.md
  B. 외환노출 법인 클러스터 (2025-12 노출 598곳, k-means, k=3~8 실루엣 최대, 최소 군집 10곳)
  C. 수출–상품 관계 (2025-12 관측 7,389곳, 집단별 보유율 lift)
그 밖의 가지는 기존 분석 결과(β 재검정, 달력 보정, 표7·포착률·비대칭 재확인)를 옮긴다.
출력: 분석결과\\insight\\insight_data.json, cluster_profile.csv, product_lift.csv
실행: py -3.11 insight_map.py   (데이터 폴더에는 쓰지 않음)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA = r"C:\test\data\processed\df_ready.csv"
BASE = r"C:\test\분석결과"
MATCH = os.path.join(BASE, "matching")
OUT = os.path.join(BASE, "insight")
SNAP = 202512                     # 스냅샷 월
WIN = (202501, 202512)            # 클러스터 특성 기간 (최근 12개월)
K_RANGE = range(3, 9)
MIN_CLUSTER = 10
MIN_NAME = 5                      # 업종 이름을 표시할 최소 법인 수
TOP_IND = 6                       # 상품 관계에 넣을 업종 수
LIFT_HI, LIFT_LO = 1.2, 0.8

HOLD = {   # 보유 항목 이름 → 계산식 (2025-12 잔액 > 0)
    "할인어음": lambda d: d["운전_할인어음잔액"],
    "무역금융": lambda d: d["운전_무역금융잔액"],
    "순수운전자금": lambda d: d["여신_운전자금대출잔액"] - d["운전_할인어음잔액"] - d["운전_무역금융잔액"],
    "시설자금": lambda d: d["여신_시설자금대출잔액"],
    "저축성예금": lambda d: d["거치식예금잔액"] + d["적립식예금잔액"],
    "투자상품": lambda d: d["수익증권잔액"] + d["신탁잔액"],
    "퇴직연금": lambda d: d["퇴직연금잔액"],
}
EXTRA = {  # 상품 관계(C)에만 추가
    "당좌대출": lambda d: d["운전_당좌대출잔액"],
    "외상매출채권담보": lambda d: d["운전_외상매출채권담보대출잔액"],
    "신용카드": lambda d: d["신용카드사용금액"],
}
CONT = {"수출비중": ("수출 비중", "높음", "낮음"), "외환규모": ("외환 규모", "큼", "작음"),
        "요구불규모": ("요구불", "큼", "작음")}


def load():
    cols = ["기준년월", "법인ID", "요구불예금잔액", "외환_수출실적금액", "외환_수입실적금액",
            "여신_운전자금대출잔액", "운전_할인어음잔액", "운전_무역금융잔액", "여신_시설자금대출잔액",
            "거치식예금잔액", "적립식예금잔액", "수익증권잔액", "신탁잔액", "퇴직연금잔액",
            "운전_당좌대출잔액", "운전_외상매출채권담보대출잔액", "신용카드사용금액"]
    d = pd.read_csv(DATA, usecols=cols)
    d = d[d["기준년월"].between(*WIN)].fillna(0)
    per = pd.read_csv(os.path.join(MATCH, "persona_firm_month.csv"), dtype={"연월": int},
                      usecols=["법인ID", "연월", "지역", "업종_대분류", "업종_중분류", "세그먼트", "페르소나"])
    per = per[per["연월"] == SNAP].fillna("")
    return d, per


def holdings(snap, items):
    return pd.DataFrame({k: (f(snap) > 0).astype(int) for k, f in items.items()}, index=snap.index)


def clusters(d, per):
    exp = per[per["세그먼트"] != "비노출"].set_index("법인ID")
    w = d[d["법인ID"].isin(exp.index)]
    agg = w.groupby("법인ID")[["외환_수출실적금액", "외환_수입실적금액"]].sum()
    snap = w[w["기준년월"] == SNAP].set_index("법인ID")
    X = pd.DataFrame(index=exp.index)
    tot = agg["외환_수출실적금액"] + agg["외환_수입실적금액"]
    X["수출비중"] = (agg["외환_수출실적금액"] / tot.replace(0, np.nan)).reindex(X.index).fillna(0)
    X["외환규모"] = np.log1p(tot.reindex(X.index).fillna(0))
    X["요구불규모"] = np.log1p(snap["요구불예금잔액"].clip(lower=0).reindex(X.index).fillna(0))
    X = X.join(holdings(snap, HOLD).reindex(X.index).fillna(0).astype(int))
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)

    scores, best = {}, None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Z)
        sizes = np.bincount(km.labels_)
        sc = float(silhouette_score(Z, km.labels_))
        scores[k] = {"실루엣": round(sc, 4), "최소군집": int(sizes.min()), "사용가능": bool(sizes.min() >= MIN_CLUSTER)}
        if sizes.min() >= MIN_CLUSTER and (best is None or sc > best[1]):
            best = (k, sc, km.labels_)
    k, sc, lab = best
    order = pd.Series(lab).value_counts().index.tolist()          # 큰 군집부터 1번
    remap = {c: i + 1 for i, c in enumerate(order)}
    X["군집"] = [remap[c] for c in lab]
    info = exp.join(X[["군집"]])

    out, rows = [], []
    for c in sorted(X["군집"].unique()):
        m = X["군집"] == c
        zc = Z[m.values].mean()
        top = zc.abs().sort_values(ascending=False).index[:2]
        name = []
        for f in top:
            if f in CONT:
                lbl, up, dn = CONT[f]
                name.append(f"{lbl} {up if zc[f] > 0 else dn}")
            else:
                name.append(f"{f} {'보유 많음' if zc[f] > 0 else '보유 적음'}")
        g = info[info["군집"] == c]
        ind = g["업종_중분류"].str.split(";").str[0].value_counts()
        prof = {
            "id": int(c), "이름": " · ".join(name), "법인수": int(m.sum()),
            "지역": g["지역"].value_counts().to_dict(),
            "세그먼트": g["세그먼트"].value_counts().to_dict(),
            "업종상위": [{"업종": i, "법인수": int(n)} for i, n in ind.items() if n >= MIN_NAME][:3],
            "수출비중_평균": round(float(X.loc[m, "수출비중"].mean()), 3),
            "외환규모_z": round(float(zc["외환규모"]), 2), "요구불규모_z": round(float(zc["요구불규모"]), 2),
            "요구불_중앙값": round(float(np.expm1(X.loc[m, "요구불규모"]).median()), 1),
            "보유율": {h: round(float(X.loc[m, h].mean()), 3) for h in HOLD},
            "특성z": {f: round(float(zc[f]), 2) for f in Z.columns},
            "페르소나": {p[:1]: int(n) for p, n in g["페르소나"].value_counts().items()},
        }
        out.append(prof)
        rows.append({"군집": c, "이름": prof["이름"], "법인수": prof["법인수"], **{f"보유_{h}": v for h, v in prof["보유율"].items()},
                     "수출비중_평균": prof["수출비중_평균"], "요구불_중앙값": prof["요구불_중앙값"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cluster_profile.csv"), index=False, encoding="utf-8-sig")
    overall = {h: round(float(X[h].mean()), 3) for h in HOLD}
    overall["수출비중_평균"] = round(float(X["수출비중"].mean()), 3)
    return {"k": int(k), "실루엣": round(sc, 4), "후보": scores, "대상": int(len(X)), "전체": overall,
            "군집": out, "특성": list(Z.columns)}


def product_lift(d, per):
    snap = d[d["기준년월"] == SNAP].set_index("법인ID")
    base = per.set_index("법인ID")
    H = holdings(snap, {**HOLD, **EXTRA}).reindex(base.index).fillna(0).astype(int)
    allr = H.mean()
    groups = [("세그먼트", s, base["세그먼트"] == s) for s in ["수출형", "수입형", "비노출"]]
    top = base["업종_대분류"].value_counts().head(TOP_IND).index
    groups += [("업종", i, base["업종_대분류"] == i) for i in top]
    out, rows = [], []
    for typ, name, m in groups:
        r = H[m.values].mean()
        lift = (r / allr.replace(0, np.nan)).fillna(0)
        out.append({"유형": typ, "이름": name, "법인수": int(m.sum()),
                    "보유율": {k: round(float(v), 3) for k, v in r.items()},
                    "lift": {k: round(float(v), 2) for k, v in lift.items()}})
        rows += [{"유형": typ, "집단": name, "상품": k, "보유율": round(float(r[k]), 4), "lift": round(float(lift[k]), 3)} for k in H.columns]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "product_lift.csv"), index=False, encoding="utf-8-sig")
    return {"대상": int(len(base)), "상품": list(H.columns), "전체보유율": {k: round(float(v), 3) for k, v in allr.items()},
            "집단": out, "기준": {"강조_이상": LIFT_HI, "강조_이하": LIFT_LO}}


def industry_exposure(per):
    g = per.groupby("업종_대분류")
    t = pd.DataFrame({"법인수": g.size(), "노출": g["세그먼트"].apply(lambda s: (s != "비노출").sum()),
                      "수출형": g["세그먼트"].apply(lambda s: (s == "수출형").sum()),
                      "수입형": g["세그먼트"].apply(lambda s: (s == "수입형").sum())})
    t = t[t["법인수"] >= 50].copy()
    t["노출비중"] = (t["노출"] / t["법인수"]).round(3)
    t = t.sort_values("노출비중", ascending=False)
    return [{"업종": i, **{k: (float(v) if k == "노출비중" else int(v)) for k, v in r.items()}} for i, r in t.iterrows()]


def main():
    os.makedirs(OUT, exist_ok=True)
    d, per = load()
    trig = pd.read_csv(os.path.join(MATCH, "trigger_stages.csv"))
    beta = pd.read_csv(os.path.join(BASE, "민영_β합검정.csv"))
    beta = beta[beta["사양"] == "주_보정충격"]
    data = {
        "생성일": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "지역수출": [{"지역": r.지역, "연월": str(r.연월), "원YoY": round(r.원YoY, 1), "보정YoY": round(r.보정YoY, 1)}
                  for r in trig.itertuples()],
        "달력": {"기울기": 4.46, "상관": 0.45, "관측": 72, "근거": "민영_포착률보정.md (72 지역×월 합동 추정)"},
        "반응": [{"h": int(r.h), "노출": round(-r.b13 * 100, 2), "하한": round(-r.ci13_hi * 100, 2),
                "상한": round(-r.ci13_lo * 100, 2), "비노출": round(-r.b1 * 100, 2),
                "유의": None if pd.isna(r.p13_holm) else bool(r.p13_holm < .05),
                "차이유의": None if pd.isna(r.p3_holm) else bool(r.p3_holm < .05)} for r in beta.itertuples()],
        "업종노출": industry_exposure(per),
        "상품관계": product_lift(d, per),
        "클러스터": clusters(d, per),
        "분석노트": [   # 판정(맞다/틀리다) 없이 무엇을 봤고 무엇이 나왔는지만 적는다
            {"이름": "외환노출 법인의 요구불 반응",
             "요약": "수출 YoY가 10%p 하락했을 때 외환노출 법인의 요구불예금 변화를 봤다. 6개월 뒤까지 −3.9%, 12개월 뒤까지 −5.0%로 나타났고, 비노출 법인과의 차이는 6~8개월 시점에서 가장 분명했다.",
             "출처": "민영_β합검정.md"},
            {"이름": "하락기와 증가기의 반응 크기",
             "요약": "수출이 깊게 떨어질 때와 늘 때 요구불 반응 크기가 다른지 비교했다. 두 시기의 반응 차이는 통계적으로 뚜렷하게 나타나지 않았다.",
             "출처": "민영_비대칭재확인.md"},
            {"이름": "은행 거래 신호와 수출 하락기",
             "요약": "요구불 감소 + 할인어음 증가 신호가 수출 하락기에 켜지는지 봤다. 영업일수 보정 뒤 하락기에 신호가 켜진 비율은 51.1%로, 평소 신호가 켜지는 비율(53.0%)과 비슷했다.",
             "출처": "민영_포착률보정.md"},
            {"이름": "전자부품 업종의 운전자금",
             "요약": "전자부품 업종의 운전자금대출이 수출·생산 충격에 반응하는지 봤다. 두 충격 모두 같은 방향의 반응이 나왔지만, 달 순서를 섞은 비교에서도 비슷한 크기가 자주 나와 해석에 주의가 필요하다.",
             "출처": "민영_표7재확인.md"},
        ],
    }
    js = json.dumps(data, ensure_ascii=False)
    assert "법인ID" not in js
    with open(os.path.join(OUT, "insight_data.json"), "w", encoding="utf-8") as f:
        f.write(js)
    c = data["클러스터"]
    print(f"저장: {OUT}\\insight_data.json ({len(js) / 1024:.0f} KB)")
    print(f"[클러스터] 대상 {c['대상']} | 선택 k={c['k']} (실루엣 {c['실루엣']}) | 후보 {c['후보']}")
    for g in c["군집"]:
        print(f"  {g['id']}. {g['이름']} — {g['법인수']}곳 | 세그먼트 {g['세그먼트']} | 페르소나 {g['페르소나']} | 업종 {[x['업종'] for x in g['업종상위']]}")
    print("[상품관계] lift ≥1.2 / ≤0.8")
    for g in data["상품관계"]["집단"]:
        hi = {k: v for k, v in g["lift"].items() if v >= LIFT_HI}
        lo = {k: v for k, v in g["lift"].items() if v <= LIFT_LO}
        print(f"  {g['이름']} ({g['법인수']}): 높음 {hi} | 낮음 {lo}")
    print("[업종 노출비중 상위]", [(x["업종"], x["노출비중"]) for x in data["업종노출"][:6]])


if __name__ == "__main__":
    main()
