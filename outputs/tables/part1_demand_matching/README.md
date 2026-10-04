# 파트 1 결과표 — 요구불 신호와 매칭 모델

허민영·신다혜 파트의 **집계 결과표**입니다. 법인 단위 값과 법인ID는 없습니다. 만든 코드는 [`src/part1_demand_matching`](../../../src/part1_demand_matching/README.md), 결과 해석은 [Don-Ddok_Docs `분석결과/파트1_요구불신호_매칭모델/`](https://github.com/Don-Ddok/Don-Ddok_Docs/tree/main/분석결과/파트1_요구불신호_매칭모델)에 있습니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `분석별/01_harmonized` | 네 계정 공통 사양 재추정. `results_all.csv`(전체), `judgment.csv`(판정), `mde.csv`(최소검출효과), `pretrend*.csv`(사전추세), `balance_*.csv`(매칭 균형), `parts/`·`parts_iter/`(계정×모형별 원 결과), `addon_multiplicity/`(부가 점검) |
| `분석별/02_did_demand_deposit` | 요구불 DID 결과표와 사전추세 |
| `분석별/03_did_loan_robust` | 대출 강건성 |
| `분석별/04_firm_shock` | 기업별 외환 충격 |
| `분석별/05_industry_shock` | 업종 충격(HS4→KSIC) |
| `분석별/06_mechanism_inflow` | 입금 경로 메커니즘 |
| `분석별/07_trade_extra_mde`, `08_trade_finance_mde` | 사전 최소검출효과 판단 |
| `분석별/09_exploratory` | 탐색 분석 로그 |
| `분석별/10_devreview` | 검산 |
| `분석별/11_insight` | 법인 군집(k=6) 프로파일, 상품 보유 배율 |
| `분석별/12_비교_파트3` | 파트 3 규칙과의 비교 |
| `분석별/13_통합분석` | 세 계정 LP, 지역×월 상관·집계 |
| `분석기록/` | 포착률 달력 보정, 비대칭 재확인, 표7 재확인, β₁+β₃ 합 검정의 결과표. `matching/`은 매칭 모델 산출 요약(페르소나 분포, 추천 상위, 트리거 단계) |
| `자금압박지수/` | 자금압박지수 시차 비교(`idea1_*`), 포착 선행시간(`idea2_*`), 업종×계정 전수 검정(`idea3_*`) |

## 공개 전 처리

- **5곳 미만인 칸은 `-1`로 가렸습니다**(대시보드 공개 데이터와 같은 규칙). 해당 파일: `분석기록/matching/persona_snapshot_202512.csv`(5칸), `분석기록/matching/recommend_demo_202507.csv`(4칸), `분석기록/matching/recommend_summary_demo_candidates.csv`(1칸), `자금압박지수/idea3_full_table.csv`(453칸). `-1`은 "1~4곳"을 뜻합니다.
- 법인ID 열이 있는 `growth_momentum_scores.csv`(7,954행)와 `step4_top20_compare.csv`는 올리지 않았습니다.
- 법인 단위 대용량 파일(`recommend_firm_month.csv`, `persona_firm_month.csv`, `df_ready.csv`)도 올리지 않았습니다.

## 읽을 때

`분석별/01_harmonized/judgment.csv`가 네 계정의 최종 판정표입니다. 요구불만 "확정"이고, 운전자금은 매칭 없는 공통 표본에서 근거 없음(착시 가능성)으로 나옵니다 — 파트 3의 매칭 결과(β₃ −0.231, p 0.094)와 함께 읽어야 합니다.
