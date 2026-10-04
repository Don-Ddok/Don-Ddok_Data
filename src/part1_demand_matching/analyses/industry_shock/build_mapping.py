# -*- coding: utf-8 -*-
"""HS4 → KSIC 10차 제조업 중분류 배분표 (결과변수·회귀 없음, 충격 계산 전 단계)

연쇄: HS 2017 6단위 → CPC 2.1 (UNSD CPC21-HS2017) → ISIC Rev.4 4단위 (UNSD isic4-cpc21) → KSIC 10차 중분류 (아래 ISIC_TO_KSIC 규칙)
HS4 배분(제안): HS4 안의 HS6를 같은 비중으로 보고, HS6 → CPC, CPC → ISIC가 여러 개면 다시 같은 비중으로 나눈다.
  → HS4마다 KSIC 중분류별 가중치(합 1). 제조업(KSIC 10~34) 밖으로 가는 몫은 "비제조"로 남긴다.
HS 2022 신설 4단위(HS 2017 표에 없음)는 품목 설명으로 수동 배정한다(아래 MANUAL, 결과를 보기 전에 고정).
출력: isic4_to_ksic.csv, hs4_ksic_weights.csv, mapping_summary.txt
"""
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
C = r"C:\test\external\industry_export\concordance"
H4 = r"C:\test\external\industry_export\hs4"
OUT = r"C:\test\outputs\industry_shock"

KSIC = {"10": "식료품 제조업", "11": "음료 제조업", "12": "담배 제조업", "13": "섬유제품 제조업; 의복 제외",
        "14": "의복, 의복액세서리 및 모피제품 제조업", "15": "가죽, 가방 및 신발 제조업", "16": "목재 및 나무제품 제조업; 가구 제외",
        "17": "펄프, 종이 및 종이제품 제조업", "18": "인쇄 및 기록매체 복제업", "19": "코크스, 연탄 및 석유정제품 제조업",
        "20": "화학 물질 및 화학제품 제조업; 의약품 제외", "21": "의료용 물질 및 의약품 제조업", "22": "고무 및 플라스틱제품 제조업",
        "23": "비금속 광물제품 제조업", "24": "1차 금속 제조업", "25": "금속 가공제품 제조업; 기계 및 가구 제외",
        "26": "전자 부품, 컴퓨터, 영상, 음향 및 통신장비 제조업", "27": "의료, 정밀, 광학 기기 및 시계 제조업", "28": "전기장비 제조업",
        "29": "기타 기계 및 장비 제조업", "30": "자동차 및 트레일러 제조업", "31": "기타 운송장비 제조업", "32": "가구 제조업",
        "33": "기타 제품 제조업", "34": "산업용 기계 및 장비 수리업"}


def isic_to_ksic(cls):
    """ISIC Rev.4 4단위 → KSIC 10차 중분류 코드 (제조업 밖이면 '비제조')"""
    d = int(cls[:2])
    if 10 <= d <= 25:
        return f"{d:02d}"
    if d == 26:
        return "27" if cls[:3] in ("265", "266", "267") else "26"      # 측정·시험·항해·시계, 전자의료, 광학 → KSIC 27
    if d == 32:
        return "27" if cls == "3250" else "33"                          # 의료·치과 기기 → KSIC 27 의료용 기기
    if 27 <= d <= 33:
        return f"{d + 1:02d}"                                           # 전기장비·기계·자동차·기타운송·가구·기타제품·수리: 번호 하나씩 밀림
    return "비제조"


# HS 2022 신설·변경 4단위 수동 배정 (HS 2017 표에 없는 것) — 품목 설명 기준
MANUAL = {"8524": ("26", "평판디스플레이 모듈 → 전자부품(ISIC 2610)"), "8485": ("29", "적층제조기계 → 기타 기계(ISIC 2829)"),
          "8807": ("31", "항공기·무인기 부분품 → 기타 운송장비(ISIC 3030)"), "8806": ("31", "무인기 → 기타 운송장비(ISIC 3030)"),
          "3827": ("20", "할로겐화 유도체 혼합물 → 화학(ISIC 2011)"), "2404": ("12", "담배·니코틴 물품 → 담배(ISIC 1200)"),
          "8549": ("비제조", "전기·전자 폐기물·스크랩 → 재생(ISIC 3830, 비제조)"), "2424": ("비제조", "이사화물 → 품목 아님")}


def main():
    a = pd.read_csv(os.path.join(C, "CPC21-HS2017.csv"), dtype=str).rename(columns={"HS 2017": "hs6", "CPC Ver. 2.1": "cpc"})
    b = pd.read_csv(os.path.join(C, "isic4-cpc21.txt"), dtype=str).rename(columns={"ISIC4code": "isic", "CPC21code": "cpc"})
    a["hs6"] = a["hs6"].str.replace(".", "", regex=False)
    a["w1"] = 1 / a.groupby("hs6")["cpc"].transform("size")                   # HS6 → CPC 여럿이면 같은 비중
    b["w2"] = 1 / b.groupby("cpc")["isic"].transform("size")                  # CPC → ISIC 여럿이면 같은 비중
    m = a.merge(b[["cpc", "isic", "w2"]], on="cpc", how="left")
    miss = m[m["isic"].isna()]
    m["ksic"] = m["isic"].map(lambda x: isic_to_ksic(x) if isinstance(x, str) else "미연결")
    m["w2"] = m["w2"].fillna(1.0)
    m["w"] = m["w1"] * m["w2"]
    m["hs4"] = m["hs6"].str[:4]
    n6 = m.groupby("hs4")["hs6"].transform("nunique")
    m["w"] = m["w"] / n6                                                       # HS4 안 HS6 같은 비중
    wt = m.groupby(["hs4", "ksic"])["w"].sum().reset_index()
    for k, (ks, note) in MANUAL.items():
        wt = pd.concat([wt, pd.DataFrame([{"hs4": k, "ksic": ks, "w": 1.0, "수동": note}])], ignore_index=True)
    chk = wt.groupby("hs4")["w"].sum()
    assert ((chk - 1).abs() < 1e-9).all(), "HS4 가중치 합이 1이 아님"
    wt["업종명"] = wt["ksic"].map(KSIC).fillna(wt["ksic"])
    wt.to_csv(os.path.join(OUT, "hs4_ksic_weights.csv"), index=False, encoding="utf-8-sig")

    # ISIC → KSIC 대응표 (UNSD 표에 나오는 제조업 ISIC 4단위 전부)
    cls = sorted(set(b["isic"]))
    it = pd.DataFrame({"ISIC4": cls})
    it["KSIC중분류"] = it["ISIC4"].map(isic_to_ksic)
    it["업종명"] = it["KSIC중분류"].map(KSIC).fillna("비제조")
    it = it[it["ISIC4"].str[:2].astype(int).between(10, 33)]
    it.to_csv(os.path.join(OUT, "isic4_to_ksic.csv"), index=False, encoding="utf-8-sig")

    # 요약: 수집된 HS4 수출이 어디로 가는지 (2022~2025 누계, 품목 수출 금액 기준 — 충격 계산 아님)
    h = pd.concat([pd.read_csv(os.path.join(H4, f), encoding="utf-8-sig", dtype={"hsSgn": str}) for f in os.listdir(H4)])
    tot = h.groupby(["지역", "hsSgn"])["expUsdAmt"].sum().reset_index().merge(wt, left_on="hsSgn", right_on="hs4", how="left")
    lost = tot[tot["ksic"].isna()]
    tot["배분"] = tot["expUsdAmt"] * tot["w"]
    by = tot.groupby(["ksic", "업종명", "지역"])["배분"].sum().unstack("지역").fillna(0)
    by = by / by.sum() * 100
    top = wt.sort_values("w", ascending=False).drop_duplicates("hs4")
    conc = tot.merge(top[["hs4", "w"]].rename(columns={"w": "최대몫"}), on="hs4")
    conc = conc.drop_duplicates(["지역", "hsSgn"])
    lines = [f"HS6 {a['hs6'].nunique():,}개, CPC→ISIC 미연결 HS6 행 {len(miss)} (CPC {miss['cpc'].nunique()}개: {sorted(miss['cpc'].unique())[:15]})",
             f"배분표 HS4 {wt['hs4'].nunique():,}개 (수동 {len(MANUAL)}개). 수집 HS4 중 배분표에 없는 코드: {sorted(lost['hsSgn'].unique()) or '없음'}",
             "HS4 수출 중 한 KSIC 중분류로 90% 이상 가는 HS4의 비중(수출 금액 기준): " + ", ".join(
                 f"{r} {(conc[(conc['지역'] == r) & (conc['최대몫'] >= 0.9)]['expUsdAmt'].sum() / conc[conc['지역'] == r]['expUsdAmt'].sum()):.3f}" for r in ["대구", "경북"]),
             "", "KSIC 중분류별 배분 수출 비중(%, 2022~2025 누계, 류 15~97):", by.round(2).to_string()]
    open(os.path.join(OUT, "mapping_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("\nISIC → KSIC 대응 (중분류별 ISIC 4단위):")
    print(it.groupby(["KSIC중분류", "업종명"])["ISIC4"].apply(lambda s: ", ".join(s)).to_string())


if __name__ == "__main__":
    main()
