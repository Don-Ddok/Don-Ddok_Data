# -*- coding: utf-8 -*-
"""summary.md: tests_log.csv에서 발견 목록·우연 가능성 목록·묶음별 집계 (새 추정 없음)"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\test\outputs\exploratory"
t = pd.read_csv(os.path.join(OUT, "tests_log.csv"))
f = t[t["FDR묶음"]].copy()
N = len(f)


def conv(x):
    if x["단위"] == "β3 차이":
        return f"β3 차이 {x['β3']:+.3f}"
    return f"{x['환산']:+.2f}{x['단위']}"


def lp(x):
    if x["단위"] == "β3 차이":
        return "– (차이 검정)"
    if pd.isna(x["LP결합p"]):
        return f"계산 불가 (공분산 비양정치, Wald {x['LP_Wald']:.2f})"
    return f"{x['LP결합p']:.3f}"


def pre(x):
    if pd.isna(x["원본방식_β3pre"]):
        return "–"
    s = f"{x['원본방식_β3pre']:+.3f} (p {x['원본방식_p']:.3f})"
    flags = []
    if x["사전추세_연장"]:
        flags.append("⚠ 사전 추세 연장")
    if x["사전추세_반대부호유의"]:
        flags.append("⚠ 반대 부호 유의")
    if not pd.isna(x["LP결합p"]) and x["LP결합p"] < 0.10:
        flags.append("⚠ LP 결합 p < 0.10")
    return s + (" " + ", ".join(flags) if flags else "")


def table(df):
    rows = ["| 묶음 | 결과변수 | 표본 | h=6 (10%p 하락 시 노출−비노출) | β3 | p | q (BH) | 사전추세 LP 결합 p | 원본 방식 β3pre | 주석 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in df.sort_values("q_BH").iterrows():
        rows.append(f"| {x['묶음']} | {x['결과변수']} | {x['표본']} | {conv(x)} | {x['β3']:+.3f} | {x['p']:.3f} | {x['q_BH']:.3f} | {lp(x)} | {pre(x)} | {x['주석'] if isinstance(x['주석'], str) else ''} |")
    return "\n".join(rows)


disc = f[f["q_BH"] < 0.10]
luck = f[(f["q_BH"] >= 0.10) & (f["p"] < 0.05)]
cnt = f.groupby("묶음").agg(검정수=("p", "size"), q010=("q_BH", lambda s: int((s < 0.10).sum())), p005=("p", lambda s: int((s < 0.05).sum())))
cnt_md = ["| 묶음 | 검정 수 | 발견 (q < 0.10) | p < 0.05 |", "|---|---|---|---|"]
names = {"A": "A 수신", "B": "B 여신", "C": "C 영업 활동", "D": "D 이질성(요구불)", "E": "E 노출 쪼개기"}
for g, x in cnt.iterrows():
    cnt_md.append(f"| {names[g]} | {x['검정수']} | {x['q010']} | {x['p005']} |")
cnt_md.append(f"| 합계 | {N} | {len(disc)} | {int((f['p'] < 0.05).sum())} |")
nonpsd = t[(t["구분"] == "곡선") & (t["h"] == 6) & t["LP_Wald"].notna() & (t["LP_Wald"] < 0)]

md = f"""# 탐색적 분석 (exploratory sweep)

> **탐색적 분석. 총 {N}개 검정, BH-FDR 적용. 기존 판정표와 별개.** 기존 사전등록 분석(harmonized, mechanism_inflow)의 판정·등급을 바꾸지 않는다.
> 사양: harmonized C1 그대로(run_common: 법인 FE + 지역×연월 FE + 노출×영업일수차, 보정YoY, 윈저 1/99, 법인·연월 이중 군집, t(G_min−1)). 결과변수·표본만 바꿨다. 판정 시차 h=6, FDR 묶음 = h=6 검정 {N}개. 곡선(h=−6~12)과 사전추세 회귀는 `tests_log.csv`에 모두 기록했고 FDR 묶음에는 넣지 않았다(기록된 회귀 행 {len(t):,}).
> q < 0.10 기준이라 발견 목록 중 약 10%는 잘못된 발견일 수 있다. 작성: 민영 (돈독) / 2026-09-29 / 코드 `explore.py`, `make_summary.py`, 0단계 `step0_candidates.py`
> 검증: 복사한 `hlib.py`로 harmonized 요구불 C1 h=6(β3 +0.363537, SE 0.113493)을 재현한 뒤 실행했다.

## 발견 (q < 0.10) — {len(disc)}개
사전추세 문제(연장, 반대 부호 유의, LP 결합 p < 0.10)가 있는 결과도 빼지 않고 ⚠로 표시만 했다.

{table(disc)}

## q ≥ 0.10이지만 p < 0.05 — "우연 가능성 큼" ({len(luck)}개)

{table(luck)}

## 묶음별 검정 수와 발견 수

{chr(10).join(cnt_md)}

## 발견을 읽는 법 (탐색적)
- **6개 중 5개는 요구불 잔액이다.** D의 4개는 이미 확정된 요구불 전체 결과(harmonized C1 −3.57%)가 큰 하위 집단(제조업, 전담 Y, 규모 하위, 고객등급 우수)에서도 보이는 것이다. 새 결과변수가 아니다. **집단 간 차이 7개는 모두 p ≥ 0.10**(가장 작은 것: 규모 하위 − 상위 +0.523, p 0.104)이라, 요구불 반응이 집단마다 다르다는 근거는 없다.
- **E: 수입만 하는 법인(509곳)에서도 요구불 차이가 −4.20%(q 0.090)다.** 수출 실적이 있는 525곳은 −2.84%(p 0.066)다. 요구불 반응이 수출 쪽에서만 나오지 않는다는 뜻이고, 메커니즘 분석("입금 감소로는 확인되지 않음")과 같은 방향이다. 수입만 하는 법인 결과는 LP 결합 p가 계산 불가라 사전추세를 판단할 수 없다.
- **요구불 밖의 유일한 발견은 체크카드 사용금액(−3.69%, q 0.090)이다.** 표본이 947곳(노출 90곳)으로 작고, LP 결합 p가 계산 불가다. q가 경계선(0.090)이라 다음 사전등록 후보로만 볼 수 있다.
- 여신(B) 8개는 발견이 없고 p < 0.05도 없다. 수신(A)은 총수신(요구불 포함)만 p < 0.05였다(q 0.131).
- **고객등급 우수:** 원본 방식 사전추세가 반대 부호이면서 유의하다(β3pre −0.646, p 0.048). harmonized 거치식처럼 평균 회귀 가능성이 있다.

## 읽을 때 주의
- **사전추세 LP 결합 p 계산 불가:** h=6 검정 중 {len(nonpsd)}개 결과변수·표본에서 LP 결합검정의 공분산(h=−6~−2, 법인·연월 이중 군집 합산)이 양정치가 아니어서 Wald가 음수로 나왔다. 해당 칸은 "계산 불가"로 적었다(p를 1로 두지 않음). 고유값 절단 같은 보정은 방법을 새로 고르는 것이라 하지 않았다. 기존 사전등록 분석(harmonized, mechanism_inflow, DID 두 개)의 Wald는 모두 양수였다.
- **D 이질성:** 요구불 표본 중 2023-01~03에 관측이 있는 7,907곳만. 업종은 제조업 / 도매·소매 / 기타로 묶었다. 집단 간 차이는 집단별 C1 회귀(완전 교차 모형과 같음)의 β3 차이이고, SE는 두 회귀의 영향 성분을 법인·연월 군집으로 합쳐 구했다. 차이 검정에는 사전추세를 따로 계산하지 않았다.
- **E 노출 쪼개기:** 수출 노출 = 수출 실적이 있는 525곳(harmonized EX와 같은 정의), 수입만 = 509곳, 각각 비노출 10,002곳과 비교.
- **결과표 주석:** 여신한도금액 — 운전자금 전체 한도가 아니라 한도대출분만일 수 있음(운전자금잔액÷한도 > 1인 관측 46%). 운전_무역금융 — 비노출 법인 47곳뿐. 총수신은 요구불을 포함한다.
- 0단계에서 제외: 운전_주택자금(노출 0곳), 운전_당좌대출(노출 17곳), 한도 소진율(정의 불분명). 카드·채널 변수는 이번 요청에 따라 C 묶음에서만 썼다.
"""
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(md)
print(md)
