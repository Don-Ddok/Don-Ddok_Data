# -*- coding: utf-8 -*-
"""0단계 (회귀 없음): 기업별 외환 실적 충격 후보의 분포 점검
1. 외환실적(수출+수입) 법인×월 분포: 실적 있는 달 비율, 법인당 중앙값, 반올림 정도
2. 충격 후보
   (a) 연속형: ln(FX_{t-2..t} 합 + 1) - ln(FX_{t-14..t-12} 합 + 1)
   (b) 더미: FX_{t-2..t} 합 <= 0.5 x (FX_{t-14..t-3} 월평균 x 3) = 1  (기준 12개월 평균 > 0일 때만 정의)
3. 역인과 점검: 충격 직전(t-3) ln(요구불+1) 수준과 충격의 상관 (상관만)
대상: 대구·경북, 2023-01~2025-12, 외환노출 법인 (df_ready exposed=1)
출력: step0_summary.txt, step0_tables.csv (집계만, 법인 ID·개별 금액 없음)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\firm_shock"
LOG = open(os.path.join(OUT, "step0_summary.txt"), "w", encoding="utf-8")
TAB = []


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


def tab(section, item, value):
    TAB.append({"구분": section, "항목": item, "값": value})


d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed", "외환_수출실적금액", "외환_수입실적금액", "요구불예금잔액"])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512) & (d["exposed"] == 1)].copy()
d = d.rename(columns={"기준년월": "ym"}).sort_values(["법인ID", "ym"]).reset_index(drop=True)
assert not d.duplicated(["법인ID", "ym"]).any()
d["FX"] = d["외환_수출실적금액"] + d["외환_수입실적금액"]
d["midx"] = (d["ym"] // 100) * 12 + d["ym"] % 100
key = pd.Index(pd.factorize(d["법인ID"])[0].astype(np.int64) * 10000 + d["midx"].to_numpy())
fid = pd.factorize(d["법인ID"])[0].astype(np.int64)


def at(arr, off):
    pos = key.get_indexer(fid * 10000 + d["midx"].to_numpy() + off)
    out = np.full(len(arr), np.nan); ok = pos >= 0; out[ok] = np.asarray(arr, float)[pos[ok]]
    return out


nf = d["법인ID"].nunique()
log(f"# 0단계 — 기업별 외환 실적 충격 후보 (회귀 없음)\n대상: 대구·경북 외환노출 법인 {nf:,}곳, 법인×월 {len(d):,}행, {d['ym'].min()}~{d['ym'].max()}")
per_firm_months = d.groupby("법인ID")["ym"].size()
log(f"법인당 관측 월: 중앙값 {per_firm_months.median():.0f}, 36개월 모두 {int((per_firm_months == 36).sum()):,}곳")

# ---------- 1. 외환 실적 분포 ----------
log("\n## 1. 외환실적(수출+수입) 법인×월 분포")
for c in ["외환_수출실적금액", "외환_수입실적금액", "FX"]:
    v = d[c]; pos = v[v > 0]
    share = (v > 0).mean()
    mf = (d.assign(p=v > 0).groupby("법인ID")["p"].sum())
    log(f"- {c}: 실적 > 0인 달 {share:.1%} / 법인별 실적 있는 달 수 중앙값 {mf.median():.0f} (p25 {mf.quantile(.25):.0f}, p75 {mf.quantile(.75):.0f}) "
        f"/ 실적 > 0인 법인 {int((mf > 0).sum()):,}곳")
    log(f"    실적 > 0인 달의 금액: p10 {pos.quantile(.1):.3g}, p25 {pos.quantile(.25):.3g}, 중앙값 {pos.median():.3g}, p75 {pos.quantile(.75):.3g}, p90 {pos.quantile(.9):.3g}")
    tab("1_분포", f"{c}_실적있는달비율", round(share, 4)); tab("1_분포", f"{c}_법인별실적월중앙값", mf.median())
    tab("1_분포", f"{c}_양수금액중앙값", pos.median())
v = d.loc[d["FX"] > 0, "FX"]
frac_int = np.isclose(v, np.round(v)).mean()
frac_01 = np.isclose(v * 10, np.round(v * 10)).mean()
vc = v.round(6).value_counts(normalize=True)
log(f"- 반올림(FX > 0인 {len(v):,}행): 정수값 {frac_int:.1%}, 소수 첫째 자리까지 {frac_01:.1%}, 서로 다른 값 {v.round(6).nunique():,}개")
log(f"    가장 흔한 값 5개(비율): " + ", ".join(f"{k:g} ({p:.1%})" for k, p in vc.head(5).items()))
log(f"    1 이하 값 비율 {(v <= 1).mean():.1%} (반올림 단위에 가까운 작은 값)")
tab("1_분포", "FX양수_정수비율", round(frac_int, 4)); tab("1_분포", "FX양수_서로다른값", v.round(6).nunique())
for c in ["외환_수출실적금액", "외환_수입실적금액"]:
    vv = d.loc[d[c] > 0, c]
    log(f"    {c} > 0: 정수값 {np.isclose(vv, np.round(vv)).mean():.1%}, 최빈값 {vv.round(6).value_counts().index[0]:g} ({vv.round(6).value_counts(normalize=True).iloc[0]:.1%})")

# ---------- 2. 충격 후보 ----------
log("\n## 2. 충격 후보")
fx = d["FX"].to_numpy(float)
S_now = fx + at(fx, -1) + at(fx, -2)
S_ly = at(fx, -12) + at(fx, -13) + at(fx, -14)
base12 = sum(at(fx, -k) for k in range(3, 15)) / 12 * 3          # t-14..t-3 월평균 x 3
d["a"] = np.log(S_now + 1) - np.log(S_ly + 1)
d["b"] = np.where(np.isnan(S_now) | np.isnan(base12), np.nan,
                  np.where(base12 > 0, (S_now <= 0.5 * base12).astype(float), np.nan))
d["b_base0"] = (base12 == 0) & ~np.isnan(S_now)
for s in ["a", "b"]:
    ok = d[s].notna()
    log(f"- ({s}) 정의되는 법인×월 {int(ok.sum()):,}, 법인 {d.loc[ok, '법인ID'].nunique():,}곳, 월 {d.loc[ok, 'ym'].nunique()}개 ({d.loc[ok, 'ym'].min()}~{d.loc[ok, 'ym'].max()})")
    tab("2_충격", f"{s}_정의되는법인월", int(ok.sum())); tab("2_충격", f"{s}_법인", d.loc[ok, "법인ID"].nunique())
a = d["a"].dropna()
both0 = ((S_now == 0) & (S_ly == 0))[d["a"].notna().to_numpy()].mean()
log(f"  (a) 분포: 정확히 0 {np.isclose(a, 0).mean():.1%} (두 구간 모두 실적 0: {both0:.1%}), 음수 {(a < 0).mean():.1%}, 양수 {(a > 0).mean():.1%}")
log(f"      분위: p1 {a.quantile(.01):+.2f}, p10 {a.quantile(.1):+.2f}, p25 {a.quantile(.25):+.2f}, 중앙값 {a.median():+.2f}, "
    f"p75 {a.quantile(.75):+.2f}, p90 {a.quantile(.9):+.2f}, p99 {a.quantile(.99):+.2f} / SD {a.std():.3f}")
one_side = (((S_now == 0) & (S_ly > 0)) | ((S_now > 0) & (S_ly == 0)))[d["a"].notna().to_numpy()].mean()
log(f"      한쪽 구간만 0 (0↔양수 전환): {one_side:.1%}")
tab("2_충격", "a_정확히0비율", round(float(np.isclose(a, 0).mean()), 4)); tab("2_충격", "a_SD", round(a.std(), 4))
tab("2_충격", "a_0양수전환비율", round(float(one_side), 4))
b = d["b"].dropna()
nb0 = int(d.loc[d["a"].notna(), "b_base0"].sum())
log(f"  (b) 발생 비율 {b.mean():.1%} ({int(b.sum()):,} / {len(b):,}), 기준 12개월 실적 0이라 정의 안 됨 {nb0:,}행")
log(f"      발생 법인 {d.loc[d['b'] == 1, '법인ID'].nunique():,}곳 / 정의되는 법인 {d.loc[d['b'].notna(), '법인ID'].nunique():,}곳")
tab("2_충격", "b_발생비율", round(b.mean(), 4)); tab("2_충격", "b_발생건", int(b.sum()))
# 변동 분해: 법인 FE, 지역×연월 FE를 뺀 뒤 남는 변동
from scipy import sparse
for s in ["a", "b"]:
    x = d.loc[d[s].notna(), ["법인ID", "사업장_시도", "ym", s]].copy()
    tot = x[s].var()
    wf = (x[s] - x.groupby("법인ID")[s].transform("mean")).var()
    # 법인 + 지역×연월 두 FE 제거 (교대투영)
    r = x[s].to_numpy(float).copy(); g1 = pd.factorize(x["법인ID"])[0]; g2 = pd.factorize(x["사업장_시도"] + x["ym"].astype(str))[0]
    for _ in range(500):
        old = r.copy()
        for g in (g1, g2):
            r -= (np.bincount(g, r) / np.bincount(g))[g]
        if np.abs(r - old).max() < 1e-10:
            break
    log(f"  ({s}) 분산: 전체 {tot:.4f}, 법인 내 {wf:.4f} ({wf / tot:.0%}), 법인·지역×연월 FE 제거 후 {r.var():.4f} ({r.var() / tot:.0%})")
    tab("2_충격", f"{s}_FE후분산비율", round(r.var() / tot, 4))
# 판정 시차별 사용 가능한 달 (충격 정의 2024-03~, 자료 끝 2025-12)
log("  시차별 추정에 쓸 수 있는 충격 월 수 (t+h <= 2025-12, t-1 관측 필요):")
months = sorted(d.loc[d["a"].notna(), "ym"].unique())
mi = lambda ym: (ym // 100) * 12 + ym % 100
endm = mi(202512)
line = []
for h in [-6, -2, 0, 3, 6, 9, 12]:
    n = sum(1 for m in months if mi(m) + max(h, 0) <= endm)
    line.append(f"h={h:+d}: {n}")
    tab("2_시차별월", f"h={h}", n)
log("    " + ", ".join(line))
# 참고 변형 (단위 점검): 금액이 0.01 단위로 작아 ln(S+1) ≈ S (로그가 아니라 금액 차이에 가까움)
okA = d["a"].notna().to_numpy()
d["a100"] = np.log(100 * S_now + 1) - np.log(100 * S_ly + 1)            # 최소 단위(0.01)=1로 바꾼 로그 차
den = (S_now + S_ly) / 2
d["dhs"] = np.where(den > 0, (S_now - S_ly) / np.where(den > 0, den, 1), 0.0)
d.loc[~okA, "dhs"] = np.nan                                               # 대칭 증가율 [-2, 2], 두 구간 0이면 0
lin = np.corrcoef(d.loc[okA, "a"], (S_now - S_ly)[okA])[0, 1]
log(f"  단위 점검: (a)와 금액 차이(S_t - S_t-12)의 상관 {lin:+.3f} (금액이 1보다 훨씬 작으면 ln(S+1) ≈ S)")
for s, nm in [("a100", "(a') ln(100·S+1) 차 [최소 단위=1]"), ("dhs", "(a'') 대칭 증가율 (S_t−S_t-12)/평균")]:
    x = d[s].dropna()
    log(f"  {nm}: p1 {x.quantile(.01):+.2f}, p10 {x.quantile(.1):+.2f}, p25 {x.quantile(.25):+.2f}, 중앙값 {x.median():+.2f}, "
        f"p75 {x.quantile(.75):+.2f}, p90 {x.quantile(.9):+.2f}, p99 {x.quantile(.99):+.2f} / SD {x.std():.3f} / |값|≥2 비율 {(x.abs() >= 2).mean():.1%}")
    tab("2_충격", f"{s}_SD", round(x.std(), 4))
    both = ~np.isnan(S_now) & ~np.isnan(S_ly) & ((S_now > 0) & (S_ly > 0))
    log(f"      두 구간 모두 양수인 행({int(both.sum()):,})만: 중앙값 {d.loc[both, s].median():+.2f}, SD {d.loc[both, s].std():.3f}")
a_b = d[["a", "b"]].dropna()
log(f"  (a)와 (b)의 상관 {a_b['a'].corr(a_b['b']):+.3f} (둘 다 정의되는 {len(a_b):,}행), (b)=1일 때 (a) 중앙값 {a_b.loc[a_b['b'] == 1, 'a'].median():+.2f}, (b)=0일 때 {a_b.loc[a_b['b'] == 0, 'a'].median():+.2f}")

# ---------- 3. 역인과 점검 ----------
log("\n## 3. 역인과 점검 — 충격 직전(t-3) 요구불 잔고 수준과 충격 (상관만, 회귀 없음)")
dd = d["요구불예금잔액"].to_numpy(float)
d["ln_dd_pre"] = at(np.log(np.clip(dd, 0, None) + 1), -3)
d["ln_dd_pre_ch"] = d["ln_dd_pre"] - at(np.log(np.clip(dd, 0, None) + 1), -15)
for s in ["a", "b"]:
    x = d[[s, "ln_dd_pre", "ln_dd_pre_ch", "법인ID"]].dropna(subset=[s, "ln_dd_pre"])
    pr = stats.pearsonr(x[s], x["ln_dd_pre"])[0]; sr = stats.spearmanr(x[s], x["ln_dd_pre"])[0]
    xw = x[s] - x.groupby("법인ID")[s].transform("mean"); lw = x["ln_dd_pre"] - x.groupby("법인ID")["ln_dd_pre"].transform("mean")
    pw = np.corrcoef(xw, lw)[0, 1]
    y = x.dropna(subset=["ln_dd_pre_ch"])
    pc = stats.spearmanr(y[s], y["ln_dd_pre_ch"])[0]
    log(f"- ({s}) n={len(x):,}: 수준 Pearson {pr:+.3f}, Spearman {sr:+.3f} / 법인 평균 뺀 뒤(법인 내) Pearson {pw:+.3f} "
        f"/ 직전 1년 요구불 변화(t-15→t-3)와 Spearman {pc:+.3f} (n={len(y):,})")
    tab("3_역인과", f"{s}_수준_Pearson", round(pr, 4)); tab("3_역인과", f"{s}_수준_Spearman", round(sr, 4))
    tab("3_역인과", f"{s}_법인내_Pearson", round(pw, 4)); tab("3_역인과", f"{s}_직전변화_Spearman", round(pc, 4))
    q = pd.qcut(x["ln_dd_pre"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    g = x.groupby(q, observed=True)[s].mean()
    log(f"    요구불 수준 5분위(1=낮음)별 충격 평균: " + ", ".join(f"Q{k} {v:+.3f}" if s == "a" else f"Q{k} {v:.1%}" for k, v in g.items()))
    for k, v in g.items():
        tab("3_역인과", f"{s}_요구불Q{k}_평균", round(v, 4))
log("\n(참고) 요구불 0인 달도 ln(0+1)=0으로 포함했다.")
pd.DataFrame(TAB).to_csv(os.path.join(OUT, "step0_tables.csv"), index=False, encoding="utf-8-sig")
LOG.close()
