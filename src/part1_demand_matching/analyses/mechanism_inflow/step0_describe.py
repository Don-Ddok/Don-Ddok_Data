# -*- coding: utf-8 -*-
"""0단계 기술 통계 (회귀 없음): 요구불 입금·출금 변화가 0인 비율, 잔액 변화 ≈ 입금 − 출금 근사, EX 노출 법인 수출 실적 분포"""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
cols = ["기준년월", "법인ID", "사업장_시도", "exposed", "요구불예금잔액", "요구불입금금액", "요구불출금금액", "외환_수출실적금액", "외환_수입실적금액"]
d = pd.read_csv(DATA, usecols=cols)
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
d = d.sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
d["midx"] = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
s = d[d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0].copy()
print(f"요구불 표본(잔액 > 0인 달 ≥ 1회): 법인 {s['법인ID'].nunique():,}, 행 {len(s):,}")
key = pd.Index(pd.factorize(s["법인ID"])[0].astype(np.int64) * 10000 + s["midx"].to_numpy())
fid = pd.factorize(s["법인ID"])[0].astype(np.int64)
def at(col, off):
    pos = key.get_indexer(fid * 10000 + s["midx"].to_numpy() + off)
    v = np.full(len(s), np.nan); ok = pos >= 0; v[ok] = s[col].to_numpy(float)[pos[ok]]; return v
for c in ["요구불입금금액", "요구불출금금액"]:
    x = s[c]
    print(f"\n[{c}] 결측 {x.isna().sum():,}, 0 이하 행 {(x <= 0).mean()*100:.1f}%, 한 번도 > 0이 아닌 법인 {(s.groupby('법인ID')[c].max() <= 0).sum():,}")
    a, b = at(c, -1), at(c, 6)
    m = ~np.isnan(a) & ~np.isnan(b)
    print(f"  t−1 → t+6 둘 다 관측 {m.sum():,}행: 변화가 정확히 0 {np.mean(a[m] == b[m])*100:.1f}% (그중 둘 다 0 {np.mean((a[m] == 0) & (b[m] == 0))*100:.1f}%p)")
    la, lb = np.log(np.clip(a[m], 0, None) + 1), np.log(np.clip(b[m], 0, None) + 1)
    print(f"  ln(+1) 변화 분위: " + ", ".join(f"p{q}={np.quantile(lb - la, q/100):+.3f}" for q in (1, 25, 50, 75, 99)))
# 잔액 변화 ≈ 입금 − 출금 (같은 달)
dB = s["요구불예금잔액"].to_numpy(float) - at("요구불예금잔액", -1)
net = (s["요구불입금금액"] - s["요구불출금금액"]).to_numpy(float)
m = ~np.isnan(dB) & ~np.isnan(net)
diff = dB[m] - net[m]
print(f"\n[잔액 변화 vs 입금 − 출금] {m.sum():,}행 (반올림 때문에 근사)")
print(f"  Pearson {np.corrcoef(dB[m], net[m])[0,1]:.3f}, Spearman {pd.Series(dB[m]).corr(pd.Series(net[m]), method='spearman'):.3f}")
print("  차이(잔액변화 − 순입금) 분위: " + ", ".join(f"p{q}={np.quantile(diff, q/100):+.3g}" for q in (1, 5, 25, 50, 75, 95, 99)))
scale = np.maximum(np.abs(s["요구불입금금액"].to_numpy(float)[m]), 1e-9)
print(f"  |차이| ÷ 입금액 중앙값 {np.median(np.abs(diff) / scale):.3f} (입금 > 0인 행 기준 {np.median(np.abs(diff[scale > 1e-9]) / scale[scale > 1e-9]):.3f})")
# EX 노출 법인 수출 실적
ex = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
e = d[d["법인ID"].isin(ex[ex].index)]
print(f"\n[EX 노출 법인] {e['법인ID'].nunique():,}곳, 행 {len(e):,}, 수출 실적 > 0인 달 비율 {(e['외환_수출실적금액'] > 0).mean()*100:.1f}%, "
      f"법인별 수출 실적 > 0인 달 수 중앙값 {e.groupby('법인ID')['외환_수출실적금액'].apply(lambda x: (x > 0).sum()).median():.0f}")
