# -*- coding: utf-8 -*-
"""성장모멘텀 파일럿 데이터 — 데모(index_template.html)에 "성장 파일럿" 화면 추가용

입력: outputs\\harmonized\\addon_multiplicity\\growth_momentum_scores.csv (탐색적 1차 실행 결과)
출력: 분석결과\\matching\\demo\\live_data.json 에 "성장파일럿" 키 추가, index.html 재생성
실행: py -3.11 build_growth_pilot_data.py  (live_data.json이 먼저 build_live_data.py로 만들어져 있어야 함)

데이터 규칙(기존 데모와 동일): 법인 ID·실제 법인명·개별 수치를 넣지 않는다. 업종 단위 집계만.
"""
import json
import os

import pandas as pd

BASE = r"C:\test\분석결과\matching\demo"
LIVE = os.path.join(BASE, "live_data.json")
TEMPLATE = os.path.join(BASE, "index_template.html")
SCORES = r"C:\test\outputs\harmonized\addon_multiplicity\growth_momentum_scores.csv"

WEIGHTS = {"거래성장": 0.5, "여신확장": 0.3, "시설투자": 0.2}
WEIGHTS_신규 = {"여신확장": 0.6, "시설투자": 0.4}
EXCL_N = 199
EXCL_PCT = 1.8
CHURN_N = 2775
NEW_N = 988
TOTAL_FIRM = 11036


def main():
    s = pd.read_csv(SCORES, index_col=0)
    ind = s.groupby("업종").agg(
        법인수=("성장모멘텀점수", "size"),
        신규수=("신규", "sum"),
        거래성장_중앙값=("거래성장", "median"),
        점수_평균=("성장모멘텀점수", "mean"),
        점수_상위10컷=("성장모멘텀점수", lambda v: v.quantile(0.9)),
        점수_최대=("성장모멘텀점수", "max"),
    ).reset_index().rename(columns={"업종": "이름"})
    ind = ind.sort_values("법인수", ascending=False)
    industries = [{
        "이름": r["이름"], "법인수": int(r["법인수"]), "신규수": int(r["신규수"]),
        "거래성장_중앙값": round(float(r["거래성장_중앙값"]), 3) if pd.notna(r["거래성장_중앙값"]) else None,
        "점수_평균": round(float(r["점수_평균"]), 3), "점수_상위10컷": round(float(r["점수_상위10컷"]), 3),
        "점수_최대": round(float(r["점수_최대"]), 3),
    } for _, r in ind.iterrows()]

    growth_pilot = {
        "meta": {
            "단계": "탐색적 1차 실행 (사전등록 전)",
            "목적": "업종 안에서 거래·여신·시설투자 성장 신호가 뚜렷한 법인을 찾아 시설자금 상담 후보로 스크리닝. 확장 성공을 예측하는 것이 아니라 지금 성장 중인 법인을 가려내는 것.",
            "가중치": WEIGHTS, "가중치_신규법인": WEIGHTS_신규,
            "비교기간": "2025년 하반기 평균 vs 2024년 하반기 평균 (거래성장은 ln(1+x) 차분)",
            "제외규칙": f"거래규모 상위 {EXCL_PCT}%({EXCL_N}곳, 연간입금 689억/400억 또는 여신한도 200억 기준) 제외 — 법적 대기업 분류 아님, 스크리닝용 대리지표",
            "전체법인": TOTAL_FIRM, "대형제외": EXCL_N, "이탈제외": CHURN_N, "신규": NEW_N,
            "분석대상": int(ind["법인수"].sum()), "추정가능업종수": len(industries),
            "한계": [
                "동행·현재상태 신호이지 확장 성공을 보장하지 않음",
                "단일 은행 데이터 기준 — iM뱅크 고객이 아닌 기업, 타행 실적은 보이지 않음",
                "업종 20곳 미만은 집계에서 제외(재식별 위험 방지)",
                "\"성장\"은 거래금액·여신한도·시설자금 대리지표일 뿐 실제 매출·이익이 아님",
                "신규 법인(988곳)은 거래성장 비교 불가 — 여신확장·시설투자만으로 환산한 참고 점수",
            ],
            "생성일": pd.Timestamp.today().strftime("%Y-%m-%d"),
        },
        "업종": industries,
    }

    with open(LIVE, encoding="utf-8") as f:
        data = json.load(f)
    data["성장파일럿"] = growth_pilot
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    assert "법인ID" not in js
    with open(LIVE, "w", encoding="utf-8") as f:
        f.write(js)
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read().replace("__DATA_JSON__", js.replace("</", "<\\/"))
    with open(os.path.join(BASE, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"저장: live_data.json에 성장파일럿 추가 ({len(js) / 1024:.0f} KB), index.html 재생성")
    print(f"업종 {len(industries)}개, 분석대상 {int(ind['법인수'].sum()):,}곳")


if __name__ == "__main__":
    main()
