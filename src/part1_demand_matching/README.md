# 파트 1 — 요구불 신호와 매칭 모델

담당: 허민영, 신다혜 · 실행 환경 **Python 3.11**

지역 수출 충격이 법인 계정(요구불·운전자금·거치식·적립식)에 전이되는지를 공통 사양으로 재추정하고, 그 결과를 상품 매칭 모델로 잇는 코드입니다.

문서와 결과 해석은 [Don-Ddok_Docs `분석결과/파트1_요구불신호_매칭모델/`](https://github.com/Don-Ddok/Don-Ddok_Docs/tree/main/분석결과/파트1_요구불신호_매칭모델)에 있습니다. 특히 **`INDEX_코드데이터지도.md`에 어떤 코드가 어떤 결과를 만드는지와 재현 순서가 정리돼 있습니다.**

## 폴더

| 폴더 | 내용 |
|---|---|
| (루트) | 매칭 모델 바깥의 분석 스크립트: 포착률 달력 보정(`financial_pressure_signals*.py`, `run_capture_*.py`), 비대칭 재확인(`asym_lp_adj.py`, `run_asym.py`), β₁+β₃ 합 검정(`beta_sum_test.py`), 통합 분석(`integrated_analysis.py` → `integrated_savings.py` → `build_integrated_pdf.py`), 인사이트 맵(`insight_map.py`), 파트3 비교(`compare_part3.py`), 영업일수 달력(`make_bizday.py`), 와일드 클러스터 부트스트랩 |
| `matching/` | 매칭 모델 1~3단계. 트리거(`trigger.py`) → 페르소나(`persona.py`) → 매칭(`matcher.py`). 규칙은 전부 `match_rules.yaml`에 있고, 데모 웹 데이터는 `build_live_data.py`가 만듭니다 |
| `analyses/<분석명>/` | 분석별 스크립트. 각 폴더의 설계(SPEC)와 결과(REPORT)는 Docs 저장소 `분석별_보고서/<같은 이름>/`에 있습니다 |
| (상품) | `imbank_product_crawler.py`, `crawl_giup.py`, `map_products.py`, `product_rules.yaml` — iM뱅크 기업 상품 수집과 매핑 |

## 분석별 폴더

| 폴더 | 분석 |
|---|---|
| `harmonized` | 네 계정 공통 사양 재추정. 공통 함수 `hlib.py`(법인 FE + 지역×연월 FE를 FWL로 제거, 법인·연월 이중 군집) |
| `did_demand_deposit` | 요구불 DID. 옛 코드 `run_old_code.py` 보존 |
| `did_loan_robust` | 대출 강건성 |
| `mechanism_inflow` | 입금 경로 메커니즘 |
| `industry_shock` | 업종 충격(HS4→KSIC 연결) |
| `firm_shock` | 기업별 외환 충격 |
| `exploratory` | 탐색 분석 |
| `trade_extra_mde`, `trade_finance_mde` | 본 분석 전 최소검출효과 판단(회귀 없음) |
| `devreview` | 검산·문구 감사 |
| `FINAL` | 최종 문서 PDF 생성 스크립트 |

## 실행 순서

`INDEX_코드데이터지도.md` 4절과 같습니다. 요약하면:

```bash
# 0) 충격 변수
py -3.11 matching/trigger.py
# 1) 파트3 패널(C2 매칭 변수에 필요) — src/part3_loan_industry의 step1_1~1_4, step3_2
# 2) 분석별
py -3.11 -u analyses/did_loan_robust/run.py    ; py -3.11 analyses/did_loan_robust/plot.py
py -3.11 -u analyses/did_demand_deposit/run.py ; py -3.11 analyses/did_demand_deposit/plot.py
cd analyses/harmonized
py -3.11 validate.py
py -3.11 -u run_harmonized.py 거치식 적립식 --industry
py -3.11 -u run_harmonized.py 운전자금 요구불 --models C1
py -3.11 assemble.py ; py -3.11 mde.py ; py -3.11 audit.py
```

필요 패키지: pandas, numpy, scipy, matplotlib, reportlab, scikit-learn, pyarrow, pyfixest(교차 확인), tabulate, linearmodels(포착률 재검증).

## 경로 주의

스크립트가 작성자 로컬 경로(`C:\test\...`)를 절대 경로로 참조합니다. 다른 컴퓨터에서 돌리려면 각 스크립트 위쪽의 경로 상수를 바꿔야 합니다. 원래 폴더 구조를 깨지 않으려고 경로는 손대지 않고 그대로 올렸습니다.

## 올리지 않은 것

| 파일 | 이유 |
|---|---|
| `df_ready.csv`(183MB), `recommend_firm_month.csv`(97MB), `persona_firm_month.csv`(40MB), 은행 원본 xlsx | 은행 원본과 법인 단위 파생 파일 (저장소 0절 원칙) |
| `growth_momentum_scores.csv`, `step4_top20_compare.csv` | 법인ID 열이 있음 |
| `YoY.ipynb`(57MB) | 실행 결과에 원자료 미리보기가 섞여 있을 수 있어 뺐습니다 |
| HS4 업종별 수출입 월별 원본 217개(40MB) | 관세청 공개 통계라 다시 받을 수 있음. 연결표만 `data/external/HS4_업종매핑_concordance/`에 둠 |
