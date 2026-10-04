# -*- coding: utf-8 -*-
"""묶음 F — 트리거의 실시간 재생과 단계 근거 (회귀 없음, 기술 통계·검산만)
디벨롭 문서 §6 묶음 F에 대응. trigger.py·match_rules.yaml(원본, 수정 안 함)은 그대로 두고,
그 산출물과 두 달력을 다시 확인한다.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_f_summary.txt"), "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


log("# 묶음 F — 트리거 실시간 재생과 단계 근거 (회귀 없음)\n")

# ---------------- 6.1/6.2 h vs k, '반응 전' 문구 ----------------
log("## 1. h(회귀 시차)와 k(트리거 경과월) 혼동 여부 — 코드 읽기 결과")
log("- `code\\matching\\trigger.py`: k = 이번 달 − T0(둔화 시작월). 회귀와 무관한 달력 개념이다.")
log("- `code\\beta_sum_test.py`: h = 회귀 기준월(요구불 잔액 관측 시점) 기준 시차. 트리거 T0와 무관하다.")
log("- `code\\matching\\build_summary_pdf.py:319`가 단계별 '근거_타이밍' 라벨에서 k 구간(관찰 k=0~2, 적기 k=3~6, 정점·유지 k=7~12)에 "
    "h 구간(h≤2, h=3~6, h=7~12)을 **그대로 대응**시킨다 — k와 h가 우연히 같은 정수 범위를 쓰지만, k는 '트리거 이후 경과월', "
    "h는 '회귀 기준월 이후 시차'로 정의가 다르다. 같은 정수라고 해서 같은 달을 가리키는 게 보장되지 않는다(트리거 T0와 회귀 기준월이 다른 개념이라서).")
log("- **이미 발견된 것:** `build_summary_pdf.py:394`에 스스로 남긴 주석 — \"관찰(k=0~2)은 '반응 전 구간'으로 표시하지만 "
    "재검정에서 h=2도 유의했다 (사전등록대로 라벨은 유지).\" 즉 **h=2가 유의했는데도 '반응 전'이라는 라벨을 그대로 썼다는 것을 "
    "이미 알고 있었다.** 이는 디벨롭 문서 §6.2가 정확히 지적한 문제다.")
log("- **권장 수정(문서 §6.2 그대로):** '반응 전 구간 (h≤2)' → '둔화 초기 관찰 구간 (h≤2, 재검정에서 h=2 유의)'. "
    "단계 구간(k=0~2 등) 자체는 사전등록된 운영 구간이라 바꾸지 않는다 — 문구만 결과와 맞춘다.")
log("- 이 수정은 `code\\matching\\build_summary_pdf.py`(3곳: 180, 319, 394행)에 있고, `분석결과\\matching\\3단계_결과.md`·`매칭모델_총정리.pdf`가 "
    "이 스크립트에서 나온다. **적용하려면 PDF도 다시 빌드해야 한다 — 이번 실행에서는 문구만 확인하고 적용은 보류했다(아래 다음 단계 참고).**\n")

# ---------------- 6.4 달력 r=0.72 재검토 ----------------
log("## 2. 영업일수 달력 'r=0.72 사실상 일치' 재검토")
a = pd.read_csv(r"C:\test\data\ext_bizday.csv")            # 자체 작성 달력
b = pd.read_csv(r"C:\test\external\Don-Ddok_Data\data\external\workdays_2021_2025.csv", encoding="utf-8-sig")
m = a.merge(b[["ym", "영업일수", "전년동월차"]], on="ym", how="inner")
m = m[m["ym"].between(202301, 202512)]                     # 팀 배포 파일과 겹치는 24개월(2023-2024, 문서가 말한 24개월)
diff_days = m["bizdays"] - m["영업일수"]
r = stats.pearsonr(m["bizdays"], m["영업일수"])[0]
log(f"- 겹치는 {len(m)}개월(2023-01~) 재계산: Pearson {r:.3f} (문서의 0.72와 대조)")
log(f"- 월별 영업일수 차이(자체달력 − 팀배포): 0인 달 {int((diff_days==0).sum())}/{len(m)}, 최대 |차| {diff_days.abs().max()}일, 평균 |차| {diff_days.abs().mean():.2f}일")
# 전년차이는 팀 파일에 이미 있고, 자체 달력은 workdays 파일처럼 전년동월차 컬럼이 없어 직접 계산
a2 = a.set_index("ym")["bizdays"]
own_diff = (a2 - a2.shift(12)).reindex(m["ym"]).to_numpy()
team_diff = m["전년동월차"].to_numpy()
ok = ~np.isnan(own_diff)
log(f"- 전년동월차(영업일수차) 비교 가능 {int(ok.sum())}개월: 부호가 다른 달 {int((np.sign(own_diff[ok]) != np.sign(team_diff[ok])).sum())}개, "
    f"평균 |차이| {np.mean(np.abs(own_diff[ok]-team_diff[ok])):.2f}일")
tri = pd.read_csv(r"C:\test\분석결과\matching\trigger_stages.csv")
sign_flip = 0
log(f"- 보정YoY 부호(둔화 여부에 직결)는 trigger_stages.csv가 이미 팀 배포 달력(workdays_2021_2025.csv)으로 계산됐으므로, "
    f"자체 달력으로 다시 계산해 부호가 바뀌는 달이 있는지는 별도로 trigger.py를 다시 안 돌리고는 확인 못 한다(원본 스크립트 미수정 원칙상 여기선 실행 안 함).")
log("- **T0·단계 차이, 추천 대상·상품 변경 수는 trigger.py를 자체 달력으로 재실행해야 나온다 — 이번 회차에서는 실행하지 않았다(원본 미수정 우선, 필요시 별도 요청).**\n")

# ---------------- 6.5 좌측절단 ----------------
log("## 3. 좌측절단 구간 — 경북 2023-01")
gb = tri[(tri["지역"] == "경북")].sort_values("연월")
log(f"- trigger_stages.csv상 경북 T0: {gb['T0'].dropna().unique().tolist()[:3]} (좌측절단 표시: {gb[gb['좌측절단']=='Y']['연월'].tolist()[:3]})")
log("- 2023-01 이전 수출 자료(2022년 YoY 계산에 필요한 2021년 자료)가 로컬에 있는지 확인:")
r22 = pd.read_csv(r"C:\test\data\ext_region.csv", encoding="utf-8-sig") if os.path.exists(r"C:\test\data\ext_region.csv") else None
if r22 is not None:
    gyears = sorted(r22[r22["region"] == "경북"]["ym"].astype(str).str[:4].unique())
    log(f"  ext_region.csv 경북 보유 연도: {gyears}")
    if "2022" in gyears or "2021" in gyears:
        log("  → 2023-01 이전 자료가 있다면 T0를 더 앞으로 검증할 수 있다. 이번 회차에서는 trigger.py 재실행 없이 자료 존재만 확인했다.")
    else:
        log("  → 2023-01 이전 자료가 없어 실제 T0(진짜 둔화 시작월)가 더 이른지 확인할 수 없다. **'경과 기간 미확정'으로 표시하는 게 맞다(현재 trigger.py도 좌측절단 'Y'로 이미 표시하고 있음 — 문서 §6.5 요청과 일치, 추가 조치 불필요).**")
LOG.close()
