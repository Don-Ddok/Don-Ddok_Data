# 파트 3 — 여신·업종 분석 코드

담당: 이성진, 전유환 · 결과 문서: [Don-Ddok_Docs `분석결과/파트3_여신업종분석/`](https://github.com/Don-Ddok/Don-Ddok_Docs)

수출 경기가 꺾일 때 대구·경북의 수출입 법인과 비슷한 조건의 비노출 법인이 운전자금 대출을 다르게 움직이는지 비교합니다. 결과를 만든 코드를 **수정 없이 그대로** 올렸습니다.

## 실행 환경

- 로컬 Windows, Python 3.13(`python`). 라이브러리는 `requirements.txt`.
- 은행 원본 데이터와 중간 파일(parquet)은 저장소에 없습니다. 코드는 분석자 로컬 폴더 구조를 기준으로 경로를 잡습니다.

```
iM Digital Banker Academy/
  ★★★2026 iM뱅크 데이터/(iM뱅크) 2026 교육용 법인 익명데이터.xlsx   ← 원본(비공개, step1_1이 읽음)
  통계 프로젝트/
    외부데이터/export_region.csv, 환율_ECOS원자료_202101_202512.csv,
    외부데이터/기준금리_ECOS원자료_202101_202512.csv, workdays_2021_2025.csv   ← 이 저장소 /data/external과 같은 파일
    여신업종분석/
      production_index_industry.csv                                  ← /data/external과 같은 파일
      내부시연/                                                       ← internal_demo_summary·combo_signal_check 결과(로컬 전용)
      단계별 분석/                                                    ← 이 폴더의 .py와 중간 parquet
```

## 실행 순서

| 순서 | 파일 | 내용 | 산출물(로컬) |
|---|---|---|---|
| 1-1 | `step1_1_extract.py` | 원본에서 필요한 열만, 대구·경북 + 금융·보험업 제외 | `step1_1_filtered.parquet` |
| 1-2 | `step1_2_firm_vars.py` | 회사 단위 값(외환노출, 수출노출, 거래기간, 규모) | `step1_2_firm_vars.parquet` |
| 1-3 | `step1_3_industry_merge.py` | 은행 업종 ↔ KOSIS 업종 연결, 생산지수 붙이기 | `step1_3_industry.parquet` |
| 1-4 | `step1_4_new_vars.py` | 대출 증감률(3·6개월), 수요환경 등 | `step1_loan_industry_panel.parquet` |
| 1-5 | `step1_5_verify.py` | 최종 패널 검증 체크리스트 | (출력만) |
| 2 | `step2_industry_offset.py` | 업종별 부호 상쇄의 원인 | (출력만) |
| 3-1~3-3 | `step3_*.py` | 공변량 점검, 성향점수매칭(1:3), 매칭 표본 회귀 | `step3_*.parquet` |
| 4-1~4-3 | `step4_*.py` | 대조군 민감도, wild cluster bootstrap, 이중강건 추정 | (출력만) |
| 5-1~5-2 | `step5_*.py` | 여신 세부 항목 점검·회귀 | `step5_2_item_results.csv` |
| 6-1~6-3 | `step6_*.py` | 분류 매트릭스, 업종별 표, 무역금융 기술통계(서술용) | `step6_2_industry_table.csv` |
| 8-1~8-4 | `step8_*.py` | 환율 통제, IPW·겹침 구간, 최소검출효과, 2023-01 관측 회사 | (출력만) |
| 8-6 | `step8_6_calendar_control.py` | 영업일수(달력) 통제 — 외부데이터 `workdays_2021_2025.csv` 필요 | `step8_6_calendar_control.csv` |
| 8-7 | `step8_7_pretrend.py` | 충격 이전 추세(사전 추세): 충격 전 6개월·3개월 변화로 같은 식 추정 | `step8_7_pretrend.csv` |
| 8-8 | `step8_8_base_rate.py` | 기준금리 통제 — 외부데이터 `기준금리_ECOS원자료_202101_202512.csv` 필요 | `step8_8_base_rate.csv` |
| 8-9 | `step8_9_month_dummies.py` | 달력월(1~12월) 더미, **사후 보조**(설계에 없던 진단) | `step8_9_month_dummies.csv` |
| 9-1 | `step9_1_downturn.py` | 수출 하락·상승 국면 분리(β₃⁻, β₃⁺) | (출력만, 9단계 요약에 포함) |
| 9-2 | `step9_2_intensity.py` | 노출 강도(연속형) 교호항 | (출력만, 9단계 요약에 포함) |
| 9-3 | `step9_3_export_only.py` | 수입만 한 회사 제외, 수출 회사만 재매칭 | (출력만, 9단계 요약에 포함) |
| 9-4 | `step9_4_balanced_panel.py` | 36개월 균형 패널 재매칭 | (출력만, 9단계 요약에 포함) |
| 공통 | `common.py` | 패널 준비, 고정효과 회귀, 이중 클러스터, 매칭 가중치 | |
| 점검 | `internal_demo_summary.py` | 대시보드 참고 신호 규칙을 실제 데이터에 적용한 집계 | `내부시연/summary.json`(로컬 전용) |
| 점검 | `internal_firms_export.py` | 대시보드 내부 시연 모드용 거래처 데이터(외환노출 1,032곳, 법인ID 앞 8자리) 내보내기. 결과 JSON은 로컬 전용이라 올리지 않음 | `내부시연/firms.json`(로컬 전용) |
| 점검 | `combo_signal_check.py` | 팀 조합 신호(요구불예금 감소 + 할인어음 증가) 개별 법인 점검. 설계·통과 기준은 머리말에 실행 전 고정 | `내부시연/combo_check.json`(로컬 전용) |

## 주 분석 기준(4단계 전에 고정)

6개월 창(h=6), 매칭 1:3, 법인·월 이중 클러스터. 판정: p<0.05 유의 / 0.05~0.10 약한 증거 / ≥0.10 근거 없음. 월 클러스터만 쓴 p값은 기간 겹침 때문에 낙관적이라 쓰지 않습니다.

9단계(9-1~9-4)는 6단계 이후에 새로 설계한 검정 4개로, 이중 클러스터 p를 Holm 보정한 값으로 판정합니다. 8-6은 8단계와 같은 강건성 점검(주 결과의 절반 기준)입니다. 설계와 해석은 Don-Ddok_Docs `분석결과/파트3_여신업종분석/06`, `08` 문서에 있습니다.

## 결과 표

`/outputs/tables/part3_loan_industry/`의 CSV는 집계 결과(법인 ID·회사 단위 값 없음)입니다.

| 파일 | 표본·모형 |
|---|---|
| `step5_2_item_results.csv` | 5단계: 매칭 표본(1:3), 세부 항목 5개 × 4구간 회귀. β3, 이중 클러스터 SE·p, Holm 보정 |
| `step6_2_industry_table.csv` | 6단계: 제조업 중분류별 불황·호황 달의 대출 증가·감소 비율(서술용, 검정 없음) |
| `step8_6_calendar_control.csv` | 8-6: 영업일수(달력) 2항 통제 전후 β₃ 비교(h=6 주 모형·이중강건, h=3 참고) |
| `step9_holm_summary.csv` | 9단계: 9-1~9-4 주 검정 β·p(이중)·Holm 보정 p·판정 |
| `step8_7_pretrend.csv` | 8-7: 충격 전 6개월·3개월 β₃, 같은 표본의 주 결과, 95% 신뢰구간 |
| `step8_8_base_rate.csv` | 8-8: 기준금리 2항 통제 전후 β₃, 금리 계수 |
| `step8_9_month_dummies.csv` | 8-9: 달력월 더미 전후 β₃(주 결과·사전 6개월), 표준오차 배수 |

## 올리지 않는 것

원본 엑셀, parquet 중간 파일, `내부시연/` 집계 JSON. 모두 은행 데이터에서 나온 파일이라 로컬에만 둡니다.
