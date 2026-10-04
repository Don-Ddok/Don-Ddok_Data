# -*- coding: utf-8 -*-
"""
얇은 셀(thin cell) 처치효과 진단 + wild cluster bootstrap 재추정 유틸리티.

배경: 계열x국면xexposed로 쪼갠 셀 회귀에서 처치 조건(exposed=1, is_down=1,
cond_no_cut_spend=1, cond_no_add_debt=1)을 모두 만족하는 "처치 법인" 수가
적으면, 일반 cluster-robust SE(CRSE)는 과소추정되어 가짜 유의성을 낸다
(few clusters 문제, Cameron & Miller 2015; MacKinnon & Webb 2018).
이 모듈은 (1) 처치 법인 수를 진단하고, (2) 필요하면 wild cluster bootstrap으로
제약형(WCR)/비제약형(WCU) p값을 재계산해 CRSE와 나란히 비교한다.

wildboottest 패키지 API 확인 결과(문서/소스/실행 재현):
  - wildboottest(model, B, cluster=None, param=None, weights_type='rademacher',
                 impose_null=True, bootstrap_type='11', seed=None, ...) -> pd.DataFrame
  - model 인자는 statsmodels의 "미적합(unfitted)" sm.OLS(y, X) 객체여야 한다.
    .fit() 한 결과(OLSResults)를 넘기면 AttributeError('exog' 없음)가 난다.
  - impose_null=True(기본) = 제약형(WCR), impose_null=False = 비제약형(WCU).
  - weights_type='webb' = Webb 6점 분포 (그 외 'rademacher'/'mammen'/'normal').
  - 반환 DataFrame은 param을 index로, columns=['statistic','p-value'].

원래 셀 회귀에서 쓴 것과 동일한 X 설계·클러스터를 그대로 넘겨야 t통계량 형태가
일치하므로, 이 모듈은 X를 재구성하지 않고 호출자가 이미 만든 (unfitted) model
객체와 클러스터 배열을 그대로 받는다.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from wildboottest.wildboottest import wildboottest

WEBB_CLUSTER_THRESHOLD = 11   # 총 클러스터 수가 이 값 이하면 Rademacher 대신 Webb
MIN_TREATED_CLUSTERS = 40     # 처치 법인 수가 이 값 미만이면 wild bootstrap 필요
P_DIFF_THRESHOLD = 0.1        # 제약형/비제약형 p값 차이가 이 값 미만이면 "신뢰 가능"


# =====================================================================
# 1단계: 처치 법인 수 진단
# =====================================================================

def diagnose_treated_clusters(cell_data: pd.DataFrame,
                               treatment_col: str = "shock_exposed_target",
                               cluster_col: str = "법인ID",
                               min_clusters: int = MIN_TREATED_CLUSTERS) -> dict:
    """셀 데이터프레임에서 처치 법인(클러스터) 수를 진단.

    n = 처치변수(treatment_col)가 0이 아닌 행의 고유 cluster_col 개수.
    n < min_clusters 이면 'wild bootstrap 필요' 플래그를 True로 반환.
    """
    treated_mask = cell_data[treatment_col] != 0
    n_treated = int(cell_data.loc[treated_mask, cluster_col].nunique())
    needs_bootstrap = n_treated < min_clusters

    print(f"[진단] 처치 법인 수(고유 {cluster_col}, {treatment_col}!=0 기준) = {n_treated}")
    if needs_bootstrap:
        print(f"  -> {min_clusters}개 미만: wild cluster bootstrap 필요")
    else:
        print(f"  -> {min_clusters}개 이상: 일반 CRSE도 참고 가능한 수준")

    return {"n_treated_clusters": n_treated, "needs_bootstrap": needs_bootstrap}


# =====================================================================
# 2단계: wild cluster bootstrap (제약형/비제약형 둘 다)
# =====================================================================

def run_wild_cluster_bootstrap(model: "sm.OLS", param: str, cluster,
                                B: int = 9999, seed: "int | None" = None) -> dict:
    """미적합(unfitted) statsmodels OLS 모델에 대해 wild cluster bootstrap을 실행.

    model: sm.OLS(y, X) 형태의 '미적합' 객체 (원래 셀 회귀와 동일한 X 설계).
           .fit() 한 결과를 넘기면 안 됨 (wildboottest가 model.exog를 직접 참조).
    param: 검정할 계수의 컬럼명 (모델 설계행렬의 컬럼명과 정확히 일치해야 함).
    cluster: 모델 표본과 동일한 행 순서/길이의 클러스터 배열(원래 셀 회귀와 동일 기준).
    B: 부트스트랩 반복 횟수(기본 9999).

    총 클러스터 수가 WEBB_CLUSTER_THRESHOLD(11) 이하면 weights_type='webb',
    그 이상이면 'rademacher'를 사용 (MacKinnon & Webb 2018 권고).

    Returns: n_clusters, weights_type, restricted(WCR)/unrestricted(WCU) 각각의
             statistic·p-value를 담은 dict.
    """
    # wildboottest 내부는 numba(njit)로 클러스터 id를 다루는데, 법인ID처럼 문자열(object)
    # 배열을 그대로 넘기면 numba 타이핑 에러가 난다(직접 재현해 확인). 그룹 구분만 유지한 채
    # 정수 코드로 변환해서 넘긴다 (실제 값 자체는 그룹핑에 쓰이지 않으므로 결과에 영향 없음).
    raw_cluster = pd.Series(np.asarray(cluster))
    n_clusters = int(raw_cluster.nunique())
    cluster_codes = pd.factorize(raw_cluster)[0]
    weights_type = "webb" if n_clusters <= WEBB_CLUSTER_THRESHOLD else "rademacher"

    print(f"[진행] wild cluster bootstrap 실행 중 (B={B}, 총 클러스터수={n_clusters}, "
          f"weights_type='{weights_type}')...")

    res_restricted = wildboottest(
        model, param=param, cluster=cluster_codes, B=B,
        impose_null=True, weights_type=weights_type, seed=seed, show=False,
    )
    res_unrestricted = wildboottest(
        model, param=param, cluster=cluster_codes, B=B,
        impose_null=False, weights_type=weights_type, seed=seed, show=False,
    )

    return {
        "n_clusters": n_clusters,
        "weights_type": weights_type,
        "restricted_stat": float(res_restricted.loc[param, "statistic"]),
        "restricted_p": float(res_restricted.loc[param, "p-value"]),
        "unrestricted_stat": float(res_unrestricted.loc[param, "statistic"]),
        "unrestricted_p": float(res_unrestricted.loc[param, "p-value"]),
    }


# =====================================================================
# 3단계: 통합 실행 (진단 -> CRSE -> wild bootstrap -> 최종 판정)
# =====================================================================

def analyze_cell_with_wild_bootstrap(cell_data: pd.DataFrame,
                                      model: "sm.OLS",
                                      fitted_result,
                                      cluster,
                                      treatment_col: str = "shock_exposed_target",
                                      cluster_col: str = "법인ID",
                                      B: int = 9999,
                                      min_clusters: int = MIN_TREATED_CLUSTERS,
                                      seed: "int | None" = 42,
                                      p_diff_threshold: float = P_DIFF_THRESHOLD) -> pd.DataFrame:
    """셀 하나를 받아: 처치 법인 수 진단 -> 일반 CRSE p값 -> (필요시) wild bootstrap
    제약형/비제약형 p값을 한 표로 출력하고, 최종 판정까지 붙여 반환.

    model: 원래 셀 회귀에 쓴 것과 동일한 (unfitted) sm.OLS(y, X) 객체.
    fitted_result: 위 model을 cov_type='cluster'로 .fit()한 결과
                   (CRSE 계수/SE/p값을 그대로 재사용하기 위함, 여기서 다시 적합하지 않음).
    cluster: model/fitted_result와 동일한 표본·행순서의 클러스터 배열.
    """
    print(f"\n[셀 분석] treatment_col='{treatment_col}', cluster_col='{cluster_col}'")

    diag = diagnose_treated_clusters(cell_data, treatment_col, cluster_col, min_clusters)

    if treatment_col not in fitted_result.params.index:
        raise KeyError(f"'{treatment_col}' 이 회귀 결과의 파라미터에 없습니다. "
                        f"model의 설계행렬 컬럼명을 확인하세요: {list(fitted_result.params.index)}")

    crse_coef = float(fitted_result.params[treatment_col])
    crse_se = float(fitted_result.bse[treatment_col])
    crse_p = float(fitted_result.pvalues[treatment_col])

    row = {
        "n_obs": int(fitted_result.nobs),
        "n_treated_clusters": diag["n_treated_clusters"],
        "wild_bootstrap_필요": diag["needs_bootstrap"],
        "coef": round(crse_coef, 6),
        "CRSE_SE": round(crse_se, 6),
        "CRSE_p": round(crse_p, 6),
    }

    if diag["needs_bootstrap"]:
        wb = run_wild_cluster_bootstrap(model, treatment_col, cluster, B=B, seed=seed)
        p_diff = abs(wb["restricted_p"] - wb["unrestricted_p"])
        verdict = "신뢰 가능" if p_diff < p_diff_threshold else "둘 다 신뢰 불가 (처치 법인 부족)"

        row.update({
            "n_clusters_total": wb["n_clusters"],
            "weights_type": wb["weights_type"],
            "WCR_p(제약형)": round(wb["restricted_p"], 6),
            "WCU_p(비제약형)": round(wb["unrestricted_p"], 6),
            "p값_차이": round(p_diff, 6),
            "최종_판정": verdict,
        })
    else:
        row.update({
            "n_clusters_total": np.nan,
            "weights_type": "-",
            "WCR_p(제약형)": np.nan,
            "WCU_p(비제약형)": np.nan,
            "p값_차이": np.nan,
            "최종_판정": "일반 CRSE로 충분 (처치 법인 수 충분)",
        })

    result_df = pd.DataFrame([row])
    print()
    print(result_df.to_string(index=False))
    return result_df


# =====================================================================
# 사용 예시 (실행하지 않음 - 참고용)
# =====================================================================
"""
import statsmodels.api as sm

# cell_data: 이미 만든 셀 표본 (exposed/is_down/cond_no_cut_spend/cond_no_add_debt
#            조건으로 필터링되어 있고, shock_exposed_target 컬럼 보유)
X_cols = ["shock_exposed_target", "다른통제변수1", "다른통제변수2"]  # 실제 셀 회귀와 동일하게
sub = cell_data.dropna(subset=["y"] + X_cols + ["법인ID"]).copy()

y = sub["y"]
X = sm.add_constant(sub[X_cols])
cluster = sub["법인ID"]

model = sm.OLS(y, X)                                            # 미적합 객체 (bootstrap용)
fitted = model.fit(cov_type="cluster", cov_kwds={"groups": cluster})  # 적합 결과 (CRSE용)

result = analyze_cell_with_wild_bootstrap(
    cell_data=sub,
    model=model,
    fitted_result=fitted,
    cluster=cluster,
    treatment_col="shock_exposed_target",
    cluster_col="법인ID",
    B=9999,
)
"""
