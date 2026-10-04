from pathlib import Path
import sys,json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,'C:/test/tmp/pdf_tools')
import pandas as pd,numpy as np
from scipy.stats import t
from reportlab.pdfbase.ttfonts import TTFont
import pymupdf
root=Path('C:/test');r=pd.read_csv(root/'outputs/harmonized/results_all.csv')
c=r[(r['모형']=='C1') & (r['업종'].isna() | (r['업종']=='전체'))].copy()
print('C1 rows',len(c),'industry',r['업종'].unique().tolist())
checks=[]
for acct,g in c.groupby('계정'):
    dof=np.minimum(g['노출법인']+g['비노출법인'],g['월'])-1
    pp=2*t.sf(np.abs(g['β3']/g['SE']),dof)
    h=g[g.h.between(1,12)].copy();ix=np.argsort(h.p.to_numpy());hp=np.empty(len(h));hp[ix]=np.minimum(1,np.maximum.accumulate((len(h)-np.arange(len(h)))*h.p.to_numpy()[ix]))
    x=g[g.h==6].iloc[0]
    checks.append({'account':acct,'p_max_diff':float(np.max(np.abs(pp-g.p))),'holm_max_diff':float(np.max(np.abs(hp-h.p_holm))),'h6_effect':float(100*np.expm1(-.1*x['β3'])),'saved_h6_effect':float(x['환산']),'significant_h':h.loc[h.p_holm<.05,'h'].tolist()})
pdf=pymupdf.open(root/'output/pdf/돈독_민영파트_최종보고서_20260930.pdf')
text=''.join(p.get_text() for p in pdf)
f=TTFont('check','C:/Windows/Fonts/malgun.ttf')
missing=sorted(set(ch for ch in text if ord(ch)>32 and ord(ch) not in f.face.charWidths))
obj={'numeric_checks':checks,'font_missing_chars':missing}
(root/'output/pdf/numeric_checks.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(obj,ensure_ascii=False))
