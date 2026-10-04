# -*- coding: utf-8 -*-
"""Build final report from saved aggregate results, without rerunning private-data analysis."""
from pathlib import Path
import sys, re, json, hashlib, subprocess, importlib.util, html
ROOT=Path('C:/test'); OUT=ROOT/'output/pdf'; TMP=ROOT/'tmp/pdfs'
sys.path.insert(0,str(ROOT/'tmp/pdf_tools'))
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.graphics.shapes import Drawing, Rect, String, Line
pdfmetrics.registerFont(TTFont('Malgun',r'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunB',r'C:/Windows/Fonts/malgunbd.ttf'))
registerFontFamily('Malgun',normal='Malgun',bold='MalgunB',italic='Malgun',boldItalic='MalgunB')
W=174*mm
NAVY='#17324D'; TEAL='#007F82'; MUTED='#526273'; LIGHT='#EFF5F7'; GOLD='#946719'
styles={}
for key,size,lead,bold in [('title',27,38,True),('sub',13,21,False),('h1',17,25,True),('h2',11.5,18,True),('p',9.6,15.5,False),('b',9.6,15.5,False),('note',8,12.5,False),('cell',8,11.5,False),('cellh',8,11.5,True)]:
    styles[key]=ParagraphStyle(key,fontName='MalgunB' if bold else 'Malgun',fontSize=size,leading=lead,textColor=colors.HexColor(NAVY if bold else '#243746'),wordWrap='CJK',spaceAfter=6 if key not in ('cell','cellh') else 0,spaceBefore=9 if key=='h2' else 0,keepWithNext=key in ('h1','h2'),splitLongWords=True)
styles['note'].textColor=colors.HexColor(MUTED)
styles['cellh'].textColor=colors.white
styles['b'].leftIndent=9
styles['h1'].spaceAfter=12
REPLACE={
 '세 계정':'네 계정',
 '사전추세는 없다.':'사전추세 검정에서 유의한 차이를 확인하지 못했다.',
 'q < 0.10 기준이라 발견 목록 중 약 10%는 잘못된 발견일 수 있다.':'BH-FDR 10%는 거짓 발견 비율의 기댓값 통제이며, 이번 발견 중 정확히 10%가 오발견이라는 뜻은 아니다.',
 'q < 0.10 기준이라 발견 중 약 10%는 잘못된 발견일 수 있다.':'BH-FDR은 적용 조건 아래 거짓 발견 비율의 기댓값을 통제한다. 이번 6개 중 정확히 10%가 오발견이라는 뜻은 아니다.',
 '공식 수출 YoY는 영업일 1일당 +4.46%p 움직인다.':'이 표본에서 영업일수 전년차 1일과 수출 YoY의 회귀 기울기는 +4.46%p다(인과효과 아님).',
 '잔액 차이는 계좌 개설·유지에서 나옴':'잔액 차이는 양수 보유 여부의 변화와 관련될 가능성이 있음',
 '월 군집 23~35개라 이중 군집 SE가 과소추정될 수 있다.':'월 군집이 23~35개로 적어 이중 군집 추론의 정밀도에 한계가 있다.',
 '각 분석 폴더(outputs\\did_loan_robust, did_demand_deposit, harmonized, mechanism_inflow, exploratory, industry_shock, firm_shock, trade_extra_mde, trade_finance_mde)는 SPEC.md를 먼저 커밋한 git 저장소이고, CHANGES.md에 에러 수정·속도 개선을 기록했다.':'분석별 git 이력을 확인했다. 주요 재검정은 SPEC 선행 커밋이 확인되며, exploratory와 사전 MDE 폴더는 별도 SPEC 선행 커밋을 일반화할 수 없다. 실제 HEAD와 작업 트리 상태는 이 보고서의 커밋 점검표에 기록했다.',
 '모든 재검정은 결과를 보기 전에 사양·판정 규칙을 문서로 고정하고 git에 커밋했다.':'주요 사후 재검정은 SPEC 선행 커밋과 결과 후속 커밋이 확인된다. 결과 열람 전 여부는 작업 기록에 따른 진술이며 git만으로 확인되지 않는다.',
 '단기투자 감소분의 절반':'단기투자 감소분의 절반',
 '사전등록 분석들은 모두 정상':'해당 탐색 이전의 사전 사양 분석은 정상',
}
def fix(t):
    t=str(t)
    for a,b in REPLACE.items():t=t.replace(a,b)
    for a in ['−','–','—','‑','–']:t=t.replace(a,'-')
    t=t.replace('ỹ','y~').replace('X̃','X~').replace('t₀.₉₇₅','t(0.975)').replace('t₀.₈₀','t(0.80)')
    return t.replace('⚠','※').replace('✓','확인').replace('✗','미충족').replace('\u00a0',' ')
def P(t,s='p'):
    p=Paragraph(fix(t),styles[s]); p.report_kind=s
    return p
def B(items):return [P('• '+t,'b') for t in items]
def T(rows,widths=None):
    n=max(len(r) for r in rows); rows=[list(r)+['']*(n-len(r)) for r in rows]
    if widths is None: widths=[1]*n
    widths=[W*x/sum(widths) for x in widths]
    pad=2 if len(rows[0])>2 and 'h=6 MDE' in str(rows[0][2]) else 3.5
    data=[[P(c,'cellh' if i==0 else 'cell') for c in r] for i,r in enumerate(rows)]
    tb=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('TEXTCOLOR',(0,0),(-1,0),colors.white),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.6,colors.HexColor(TEAL)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6F8')]),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#D9E2E8')),('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
    for cell in data[0]:cell.style=ParagraphStyle('th',parent=styles['cellh'],textColor=colors.white)
    tb.spaceAfter=9
    return tb

def inline(txt):
    txt=fix(txt)
    txt=re.sub(r'!\[([^\]]*)\]\([^)]*\)',r'[그림: \1]',txt)
    txt=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'§LINK'+str(len(LINKS))+'§' if not LINKS.append((m.group(1),m.group(2))) else '',txt)
    txt=html.escape(txt)
    txt=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',txt)
    txt=re.sub(r'`([^`]+)`',r'\1',txt)
    for i,(label,url) in enumerate(LINKS):
        txt=txt.replace('§LINK'+str(i+1)+'§',f'<link href="{html.escape(url,quote=True)}" color="{TEAL}">{html.escape(label)}</link>')
    return txt
LINKS=[]
def md_flow(path,appendix=False):
    lines=Path(path).read_text(encoding='utf-8-sig').splitlines(); out=[]; i=0; code=False
    while i<len(lines):
        l=lines[i].strip(); i+=1
        if l.startswith('```'):code=not code;continue
        if not l:continue
        if code:out.append(P(html.escape(fix(l)),'note'));continue
        if l=='---':
            if not appendix:out.append(PageBreak())
            continue
        if l.startswith('|'):
            block=[l]
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i].strip());i+=1
            rows=[]
            for bl in block:
                vals=re.split(r'(?<!\\)\|',bl.strip('|'))
                if all(re.fullmatch(r'\s*:?-+:?\s*',v) for v in vals):continue
                rows.append([inline(v.strip().replace('\\|','|')) for v in vals])
            nc=max(map(len,rows));rows=[r+['']*(nc-len(r)) for r in rows]
            if nc>8:
                for start in range(1,nc,6):
                    cols=[0]+list(range(start,min(start+6,nc)))
                    out.append(P('표 열 묶음 '+str((start-1)//6+1)+' (첫 열 기준으로 연결)','note'))
                    out.append(T([[r[c] for c in cols] for r in rows]))
            else:
                weights=[max(10,min(55,max(len(re.sub('<[^>]+>','',r[j])) for r in rows))) for j in range(nc)]
                out.append(T(rows,weights))
            continue
        if l.startswith('#'):
            level=len(l)-len(l.lstrip('#')); txt=l[level:].strip()
            kind='h1' if level==1 and not appendix else 'h2'
            out.append(P(inline(txt),kind));continue
        if l.startswith('>'):out.append(P(inline(l.lstrip('> ')),'note'));continue
        if re.match(r'^[-*] ',l):out.append(P('• '+inline(l[2:]),'b'));continue
        out.append(P(inline(l),'p'))
    return out

class ReportDoc(SimpleDocTemplate):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.heading_count=0;self.heading_records=[]
    def beforeDocument(self):self.heading_count=0;self.heading_records=[]
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and getattr(f,'report_kind','')=='h1':
            txt=f.getPlainText();self.heading_count+=1;key='section'+str(self.heading_count)
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(txt,key,level=0,closed=False)
            self.notify('TOCEntry',(0,txt,self.page,key));self.heading_records.append({'title':txt,'page':self.page})

def footer(c,d):
    if d.page==1:return
    c.saveState();c.setStrokeColor(colors.HexColor('#CBD9E2'));c.setLineWidth(.4);c.line(18*mm,17*mm,192*mm,17*mm)
    c.setFont('Malgun',7.2);c.setFillColor(colors.HexColor(MUTED))
    c.drawString(18*mm,12*mm,'돈독 | 지역 수출과 법인 은행 거래 | 2026.09.30')
    c.drawRightString(192*mm,12*mm,str(d.page));c.restoreState()

def git(repo,*args):
    p=subprocess.run(['git','-c','safe.directory='+repo.as_posix(),'-c','core.excludesFile=C:/test/tmp/pdfs/empty_gitignore','-c','core.quotepath=false','-C',str(repo),*args],capture_output=True,encoding='utf-8',errors='replace')
    if p.returncode:raise RuntimeError(p.stderr)
    return p.stdout.strip()

def manifest():
    repos=[]
    for p in sorted((ROOT/'outputs').iterdir()):
        if not (p/'.git').exists():continue
        repos.append({'repo':p.name,'head':git(p,'rev-parse','HEAD'),'status':git(p,'status','--short'),'log':git(p,'log','--format=%h | %ad | %s','--date=iso-strict','-8'),'spec_history':git(p,'log','--reverse','--format=%h | %ad | %s','--date=iso-strict','--','SPEC.md','SPEC_IMPORT.md')})
    paths=[ROOT/'outputs/FINAL/build_final_pdf.py',ROOT/'outputs/FINAL/INDEX.md']
    paths.extend((ROOT/'분석결과').glob('*.md'))
    paths.extend((ROOT/'분석결과/matching').glob('*.md'))
    paths.extend((ROOT/'output/pdf').glob('*.md'))
    for r in repos:
        paths.extend((ROOT/'outputs'/r['repo']).glob('*.md'));paths.extend((ROOT/'outputs'/r['repo']).glob('*.csv'))
    files=[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(set(paths))]
    obj={'report_date':'2026-09-30','scope':'Saved aggregate results and documents; no re-estimation','repos':repos,'files':files}
    (OUT/'source_manifest.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');return obj

def official_chart():
    d=Drawing(W,145)
    labels=['전체','대기업','중견기업','중소기업'];vals=[7.5,9.4,4.9,3.1]
    for i,(label,v) in enumerate(zip(labels,vals)):
        y=118-i*31;d.add(String(0,y,label,fontName='Malgun',fontSize=10,fillColor=colors.HexColor(NAVY)))
        d.add(Rect(85,y-3,v*29,16,fillColor=colors.HexColor(TEAL if i==3 else NAVY),strokeColor=None))
        d.add(String(93+v*29,y,f'-{v:.1f}%',fontName='MalgunB',fontSize=11,fillColor=colors.HexColor(NAVY)))
    return d

def capture_original():
    spec=importlib.util.spec_from_file_location('existing_report',ROOT/'outputs/FINAL/build_final_pdf.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    captured=[]
    class Capture:
        def __init__(self,*a,**kw):pass
        def build(self,story,**kw):captured.extend(story)
    mod.P=P;mod.B=B;mod.T=T;mod.SimpleDocTemplate=Capture;mod.OUT='(captured in memory; original PDF preserved)';mod.main()
    # Replace old cover with our current executive summary; retain all numbered analytical sections.
    idx=next(i for i,f in enumerate(captured) if isinstance(f,Paragraph) and f.getPlainText().startswith('1. 데이터'))
    result=captured[idx:]
    # Keep long numbered sections distinct; avoid a section title appended below a large previous table.
    rebuilt=[]
    for f in result:
        if isinstance(f,Paragraph) and f.getPlainText().startswith('9. 기업별') and rebuilt and not isinstance(rebuilt[-1],PageBreak):rebuilt.append(PageBreak())
        rebuilt.append(f)
    return rebuilt

def main():
    OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    m=manifest();story=[]
    story += [Spacer(1,22*mm),P('지역 수출과<br/>법인 은행 거래','title'),P('요구불예금의 동행 관계와<br/>기업별 모니터링을 위한 실무 제언','sub'),Spacer(1,10*mm),P('돈독 · 민영 파트 최종 보고서','h2'),P('iM뱅크 디지털뱅커아카데미 3조<br/>2026년 9월 30일 · 코드·MD·집계 결과 및 커밋 점검 반영','p'),Spacer(1,16*mm),T([['분석 범위','핵심 관찰'],['대구·경북 법인 11,036곳\n2023.01-2025.12','지역 수출 YoY 10%p 하락과 함께 외환노출 법인의 요구불 잔액이 6개월 시차에서 비노출 대비 약 3.57% 낮게 나타남.'],['최종 문서화 범위','공통 사양 · 강건성 · 외부 통계 연결 · 기업별 외환 실적 · MDE · 발표 구성 · 문헌 · 커밋 기록']],[57,117]),Spacer(1,7*mm),P('해석 원칙: 동행 관계와 인과 경로를 구분한다. 평균 반응과 개별 기업 선별력을 구분한다.','sub'),P('원자료와 개별 법인 정보는 수록하지 않고 집계 결과만 사용했다. 저장된 결과를 대조해 문서화했으며, 이번 제작에서 원자료 회귀를 새로 실행하지 않았다.','note'),PageBreak()]
    story.append(P('읽는 순서와 목차','h1'))
    story.append(P('앞부분은 발표와 의사결정용 요약, 1-11절은 분석 결과, 부록은 세부 검정과 재현 기록이다. PDF 책갈피와 목차에서 해당 절로 이동할 수 있다.','p'))
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='Malgun',fontSize=8.5,leading=11.5,spaceBefore=1,spaceAfter=0,textColor=colors.HexColor(NAVY),wordWrap='CJK')];story.append(toc);story.append(PageBreak())
    story+=md_flow(OUT/'presentation_notes.md')
    story.append(PageBreak());story.append(P('국내 공식 통계: 규모별 수출 흐름은 다르다','h1'));story.append(official_chart())
    story.append(P('2023년 수출 증감률. 출처: 통계청·관세청, 2023년 기업특성별 무역통계(잠정), 요약 2쪽(파일 8쪽) [R1]. 감소율 크기 9.4/3.1은 약 3.0배다.','note'))
    story+=md_flow(OUT/'reference_notes.md')
    story.append(PageBreak());story+=capture_original()
    story.append(PageBreak());story+=md_flow(OUT/'implementation_notes.md')
    story.append(PageBreak());story.append(P('커밋 점검 및 최종본의 기준','h1'))
    story.append(P('작업 시작 시점의 분석 저장소 10개를 확인했다. 분석 결과 저장소 9개는 작업 트리에 변경이 없었고, FINAL에는 INDEX.md·build_final_pdf.py·기존 PDF의 미커밋 수정이 있었다. 본 보고서는 커밋본뿐 아니라 그 최신 작업본을 읽었다.','p'))
    rows=[['저장소','HEAD','사양 선행 기록 / 상태']]
    for r in m['repos']:
        sh=r['spec_history'].splitlines();specnote=(' / '.join(x.split(' | ')[0] for x in sh)+' (SPEC 이력)') if sh else '별도 SPEC 선행 커밋 확인 안 됨'
        rows.append([r['repo'],r['head'][:7],specnote+' · '+('미커밋 수정 있음' if r['status'] else '변경 없음')])
    story.append(T(rows,[43,22,109]))
    story.append(P('git은 사양 문서가 결과보다 먼저 커밋됐는지 확인하는 근거다. 실제 결과 열람 시점이나 외부 사전등록을 증명하지는 않는다. exploratory는 탐색 분석이고 MDE 폴더는 회귀 전 실행 가능성 판단 자료다.','note'))
    story.append(P('팀 참고 저장소 external/Don-Ddok_Data도 확인했다. HEAD는 7c8c549(2026-09-29)이며 작업 트리에 변경이 없었다.','note'))
    story.append(P('저장된 C1 72개 결과의 계수·표준오차에서 p값과 계정별 Holm 보정을 독립적으로 재계산했다. 저장값과의 최대 차이는 2×10^-15 미만이며, 요구불의 Holm 유의 시차 6·7·8·11이 일치했다. 원자료 회귀를 새로 추정한 검증은 아니다.','note'))
    story.append(P('최종 문서화에서 바로잡은 표현','h2'))
    story+=B(['외부 통계 연결 검정은 지역 수출·업종 수출·업종 수입의 3종으로 셌다. 기업별 외환 실적 분석은 별도의 네 번째 점검이다.','C1은 요구불·운전자금·거치식·적립식 네 계정을 비교한다. 계정별 표본과 단위는 따로 표시했다.','MDE보다 작은 추정치라는 사실만으로 효과의 부재나 과대추정을 판정하지 않는다. 신뢰구간·다중검정·사전추세를 함께 읽는다.','BH-FDR 10%는 반복 적용에서 거짓 발견 비율의 기댓값에 관한 통제다. 이번 발견 6개 중 정확히 10%가 틀렸다는 의미가 아니다.','해외 문헌의 small firms는 멕시코 상장기업 내 상대적 규모다. 국내 법정 중소기업 표본과 동일시하지 않았고, 현지통화는 원화가 아닌 페소로 적었다.'])
    story.append(P('재현','h2'));story.append(P('실행: python output/pdf/build_report.py. source_manifest.json에 입력 문서·집계 CSV의 SHA-256 및 저장소 이력을 남겼다. 이번 제작 파일은 output/pdf에 별도로 저장하며 기존 분석 코드·데이터·PDF는 보존했다.','p'))
    appendices=[
      ('부록 A. 공통 사양의 전체 판정·강건성·MDE','outputs/harmonized/REPORT.md'),
      ('부록 B. 업종 수출 충격의 세부 검정','outputs/industry_shock/REPORT.md'),
      ('부록 C. 업종 수입 충격의 세부 검정','outputs/industry_shock/REPORT_IMPORT.md'),
      ('부록 D. 기업별 외환 실적 충격의 세부 검정','outputs/firm_shock/REPORT.md'),
      ('부록 E. 상거래 결제 금융의 사전 검정력','outputs/trade_finance_mde/prospect.md'),
      ('부록 F. 시차별 사전 검정력과 계산 불가 처리','outputs/trade_extra_mde/prospect_by_h.md'),
      ('부록 G. 영업일수 보정과 신호 평가','분석결과/민영_포착률보정.md'),
      ('부록 H. 노출 법인 전체 반응과 집단 간 차이','분석결과/민영_β합검정.md'),
      ('부록 I. 입출금 메커니즘과 외부 충격 연결','outputs/mechanism_inflow/REPORT.md'),
      ('부록 J. 전체 탐색 검정의 요약','outputs/exploratory/summary.md'),
    ]
    for title,rel in appendices:
        story += [PageBreak(),P(title,'h1'),P('기록 출처: '+rel+' · 원문 수치와 분석 판정 유지. 긴 표는 열 묶음으로 나누었으며, 해석 보완은 본문 기준으로 읽는다.','note')]
        story+=md_flow(ROOT/rel,appendix=True)
    story+=[PageBreak(),P('참고문헌과 링크','h1')];story+=md_flow(OUT/'references.md',appendix=False)
    doc=ReportDoc(str(OUT/'돈독_민영파트_최종보고서_20260930.pdf'),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=17*mm,bottomMargin=23*mm,title='지역 수출과 법인 은행 거래 | 돈독 최종 보고서',author='민영 (돈독)',pageCompression=1)
    doc.multiBuild(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'toc.json').write_text(json.dumps(doc.heading_records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL',OUT/'돈독_민영파트_최종보고서_20260930.pdf')
if __name__=='__main__':main()





