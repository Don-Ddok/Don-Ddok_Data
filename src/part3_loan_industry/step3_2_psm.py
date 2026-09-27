# -*- coding: utf-8 -*-
"""
3단계-2: 성향점수매칭(PSM) 재실행 — 이전 기준값 vs 새 기준값 비교
  처치: 외환노출(수출 또는 수입 실적 1회 이상)
  표본: 운전자금 보유이력 회사 9,528개
  [이전] 업종 + 첫 달 등급 + 규모분위(수신+여신 36개월 평균) + 거래기간
  [새  ] 업종 + 첫 달 등급 + 첫 3개월 수신(로그) + 첫 3개월 운전자금(로그) + 거래기간
  매칭: 로지스틱 회귀 성향점수, 1:3 최근접이웃(복원), 캘리퍼 없음(캘리퍼는 4단계)
입력: step1_loan_industry_panel.parquet
출력: step3_psm_matched.parquet (새 기준값의 매칭 쌍 — 4단계에서 읽음)
      step3_propensity.parquet (전체 회사 성향점수·기준값 — 4단계 재매칭·이중강건용)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
K = 3             # 노출 회사 1곳당 대조 회사 3곳
MIN_IND = 20      # 회사 20개 미만 업종은 '기타'로 묶음 (추정 불안정 방지)

df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
df = df[df["운전자금_보유이력"] == 1].sort_values(["법인ID", "ym"])
df["수신합"] = df[["요구불예금잔액", "거치식예금잔액", "적립식예금잔액"]].sum(axis=1)
df["순번"] = df.groupby("법인ID").cumcount()
early = df[df["순번"] < 3].groupby("법인ID")[["수신합", "여신_운전자금대출잔액"]].mean()

firm = df.groupby("법인ID").agg(외환노출=("외환노출", "first"), 거래기간=("거래기간", "first"),
                               규모분위=("규모분위", "first"), 업종=("업종_중분류", "first"),
                               등급=("법인_고객등급", "first"))   # 정렬돼 있으므로 first = 첫 달
firm["규모분위"] = firm["규모분위"].astype(int)
firm["log_초기수신"] = np.log1p(early["수신합"])
firm["log_초기운전"] = np.log1p(early["여신_운전자금대출잔액"])
cnt = firm["업종"].value_counts()
firm["업종"] = firm["업종"].where(firm["업종"].map(cnt) >= MIN_IND, "기타")
firm["최우수"] = (firm["등급"] == "최우수").astype(int)
firm["우수"] = (firm["등급"] == "우수").astype(int)

ind = pd.get_dummies(firm["업종"], prefix="업종", drop_first=True, dtype=int)
firm = firm.join(ind)
IND_COLS = list(ind.columns)
T = firm["외환노출"].to_numpy() == 1
print(f"표본 {len(firm):,}개 (노출 {T.sum():,} / 비노출 {(~T).sum():,}), 업종 그룹 {firm['업종'].nunique()}개")

SPECS = {
    "이전": ["최우수", "우수", "규모분위", "거래기간"] + IND_COLS,
    "새": ["최우수", "우수", "log_초기수신", "log_초기운전", "거래기간"] + IND_COLS,
}
# 균형 점검은 두 방식 모두 같은 변수 목록으로 (서로의 기준값에서도 맞는지 확인)
CHECK = ["규모분위", "log_초기수신", "log_초기운전", "최우수", "우수", "거래기간"]


def smd_table(matched_t, matched_ctrl_w):
    """매칭 전·후 SMD. 분모는 매칭 전 두 집단 분산 평균(관례)"""
    rows = []
    for c in CHECK + IND_COLS:
        x = firm[c].to_numpy(float)
        xt, xc = x[T], x[~T]
        sd = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
        before = (xt.mean() - xc.mean()) / sd
        after = (x[matched_t].mean() - np.average(x, weights=matched_ctrl_w)) / sd
        rows.append((c, before, after))
    return pd.DataFrame(rows, columns=["변수", "매칭전", "매칭후"]).set_index("변수")


results = {}
for name, cols in SPECS.items():
    X = firm[cols].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)                 # 표준화(수렴 안정용, 결과엔 영향 없음)
    model = LogisticRegression(C=np.inf, max_iter=5000).fit(X, T)
    ps = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(T, ps)

    # 공통지지: 대조군 성향점수 범위 밖의 노출 회사는 제외
    lo, hi = ps[~T].min(), ps[~T].max()
    matched_t = T & (ps >= lo) & (ps <= hi)
    ctrl_idx = np.where(~T)[0]
    nn = NearestNeighbors(n_neighbors=K).fit(ps[ctrl_idx].reshape(-1, 1))
    dist, pos = nn.kneighbors(ps[matched_t].reshape(-1, 1))
    picked = ctrl_idx[pos]                          # (노출 회사 수 × 3)

    w = np.zeros(len(firm))                         # 대조 회사 가중치 = 뽑힌 횟수
    np.add.at(w, picked.ravel(), 1)
    tab = smd_table(matched_t, w)

    print(f"\n{'='*20} [{name} 기준값] {'='*20}")
    print(f"AUC {auc:.3f} | 공통지지 밖 노출 회사 {T.sum() - matched_t.sum()}개 "
          f"| 성향점수 차이 평균 {dist.mean():.4f}, 최대 {dist.max():.4f}")
    print(f"뽑힌 대조 회사: 서로 다른 {int((w > 0).sum()):,}곳 (최다 중복 {int(w.max())}회)")
    show = tab.loc[CHECK].copy()
    show["판정"] = np.where(show["매칭후"].abs() < 0.1, "통과", "미달")
    print(show.round(3).to_string())
    ia = tab.loc[IND_COLS, "매칭후"].abs()
    print(f"업종 더미 {len(IND_COLS)}개: 매칭후 |SMD| 최대 {ia.max():.3f}, 0.1 초과 {int((ia > 0.1).sum())}개")
    results[name] = dict(ps=ps, matched_t=matched_t, picked=picked, dist=dist)

# ── 새 기준값 매칭 결과 저장 (4단계에서 캘리퍼·이중강건·민감도 분석에 사용) ──
r = results["새"]
ids = firm.index.to_numpy()
t_ids = ids[r["matched_t"]]
out = pd.DataFrame({
    "법인ID_노출": np.repeat(t_ids, K),
    "법인ID_대조": ids[r["picked"]].ravel(),
    "순위": np.tile(np.arange(1, K + 1), len(t_ids)),
    "성향점수차": r["dist"].ravel(),
})
ps_all = pd.Series(r["ps"], index=ids, name="성향점수")
out["성향점수_노출"] = out["법인ID_노출"].map(ps_all)
out.to_parquet(HERE / "step3_psm_matched.parquet", index=False)
print(f"\n저장: step3_psm_matched.parquet ({len(out):,}행 = 노출 {len(t_ids):,}곳 × {K})")

# 전체 회사 성향점수 + 매칭 기준값 (4단계 재매칭·이중강건에서 사용)
keep = ["외환노출", "log_초기수신", "log_초기운전", "최우수", "우수", "거래기간", "업종"]
firm[keep].assign(성향점수=r["ps"]).reset_index().to_parquet(HERE / "step3_propensity.parquet", index=False)
print(f"저장: step3_propensity.parquet ({len(firm):,}개 회사)")
