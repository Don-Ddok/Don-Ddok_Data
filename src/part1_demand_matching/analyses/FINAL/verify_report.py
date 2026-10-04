import sys,json,re,math
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,'C:/test/tmp/pdf_tools')
import pymupdf as fitz
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('C:/test'); out=ROOT/'output/pdf'; tmp=ROOT/'tmp/pdfs'; pdf=out/'돈독_민영파트_최종보고서_20260930.pdf'
doc=fitz.open(pdf); records=[]; problems=[]
for i,page in enumerate(doc):
    text=page.get_text(); blocks=page.get_text('dict')['blocks']; spans=[s for b in blocks if 'lines' in b for l in b['lines'] for s in l['spans']]
    bad=[s['text'] for s in spans if s['bbox'][0]<35 or s['bbox'][2]>page.rect.width-35 or s['bbox'][1]<25 or s['bbox'][3]>page.rect.height-22]
    if bad:problems.append({'page':i+1,'bounds':bad})
    if '\ufffd' in text or '§LINK' in text:problems.append({'page':i+1,'bad_text':True})
    content=[s for s in spans if s['bbox'][1]<page.rect.height-55]
    records.append({'page':i+1,'chars':len(text),'first':text[:100],'content_bottom':max((s['bbox'][3] for s in content),default=0),'images':len(page.get_images())})
    page.get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(tmp/f'page-{i+1:03d}.png')
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
for first in range(0,len(doc),12):
    sheet=Image.new('RGB',(1200,1660),'#D6DEE4');dr=ImageDraw.Draw(sheet)
    for j,i in enumerate(range(first,min(first+12,len(doc)))):
        im=Image.open(tmp/f'page-{i+1:03d}.png');im.thumbnail((280,375));x=12+(j%4)*298;y=30+(j//4)*540
        sheet.paste(im,(x,y));dr.text((x,y+385),f'Page {i+1}',fill='black',font=font)
    sheet.save(tmp/f'contact-{first//12+1:02d}.jpg')
# Recompute selected saved-result reporting arithmetic without loading private raw data.
import pandas as pd,numpy as np
from scipy.stats import t
r=pd.read_csv(ROOT/'outputs/harmonized/results_all.csv');checks={}
checks['result_columns']=list(r.columns)
checks['toc_entries']=len(doc.get_toc());checks['external_links']=sum(1 for p in doc for l in p.get_links() if l.get('uri'))
report={'pages':len(doc),'bytes':pdf.stat().st_size,'problems':problems,'pages_detail':records,'checks':checks}
(out/'qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(tmp/'report_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+p.get_text() for i,p in enumerate(doc)),encoding='utf-8')
print(json.dumps({'pages':len(doc),'bounds_or_text_problems':problems,'checks':checks,'sparse_pages':[r for r in records if r['content_bottom']<250 and r['page']>1]},ensure_ascii=False))

