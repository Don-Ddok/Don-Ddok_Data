"""iM뱅크 기업(법인) 금융상품 수집기 — imbank_product_crawler.py(최신본)를 복사해 새로 만든 파일

대상 (사전 확정):
  - 기업 대출 목록 13개(금융상품몰 > 대출 > 기업상품)의 상품 상세 페이지
  - 외환 업무 안내 페이지 8개(수출입·신용장·무역금융)
  - 기업 예금, 환위험 상품은 제외

원본 크롤러와 다른 점:
  - 대상 페이지는 전부 정적 HTML이라 Selenium 대신 requests + BeautifulSoup만 쓴다.
  - 링크를 넓게 따라가지 않고, 정해진 목록 페이지와 그 목록에 나온 상세 페이지만 연다.
  - 요청 간격은 REQUEST_INTERVAL(3초 이상)이다. robots.txt(2026-09-28 확인)에는 Disallow 규칙이 없다.

결과: 실행 폴더의 기업상품.csv (utf-8-sig)
  - 상세 페이지에 해당 제목(라벨)이 없어서 못 읽은 항목은 "파싱실패"로 둔다 (추측해서 채우지 않음).
  - 외환 안내 페이지에서 금리·한도 등이 없으면 빈 칸으로 둔다.

실행:
  pip install requests beautifulsoup4 pandas
  python crawl_giup.py                    # 결과: ./기업상품.csv
  python crawl_giup.py --cache DIR        # 받은 HTML을 DIR에 저장해 두고 재실행 때 다시 요청하지 않음
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.imbank.co.kr"
REQUEST_INTERVAL = 3.2   # 초. 3초 이상 유지
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-crawler; contact: team DonDok)"}
FAIL = "파싱실패"

# (출처목록 이름, 목록 URL). 단일 상품 폴더는 index.html이 상세 페이지로 넘어간다(meta refresh).
LOAN_LISTS = [
    ("운전자금", "/cms/fnm/loan/product/giup/01/sda_41211/index.html"),
    ("시설자금", "/cms/fnm/loan/product/giup/01/sda_41212/index.html"),
    ("B2B전자결제", "/cms/fnm/loan/product/giup/01/sda_41213/index.html"),
    ("기술우수_창업기업", "/cms/fnm/loan/product/giup/01/sda_41214/index.html"),
    ("지급보증", "/cms/fnm/loan/product/giup/01/sda_41215/index.html"),
    ("소상공인", "/cms/fnm/loan/product/giup/01/sda_41216/index.html"),
    ("추천자금", "/cms/fnm/loan/product/giup/02/sda_41221/index.html"),
    ("정책자금", "/cms/fnm/loan/product/giup/02/sda_41222/index.html"),
    ("DGB희망키움특별보증", "/cms/fnm/loan/product/giup/02/sda_41223/index.html"),
    ("골목상점가특별보증", "/cms/fnm/loan/product/giup/02/sda_41224/index.html"),
    ("DGB청년드림", "/cms/fnm/loan/product/giup/02/sda_41225/index.html"),
    ("신용보증기금보증서담보", "/cms/fnm/loan/product/giup/02/sda_41226/index.html"),
    ("국세성실납세사업자특별", "/cms/fnm/loan/product/giup/02/sda_41227/index.html"),
]
GUARANTEE_LISTS = {"지급보증"}   # 유형 = 보증

FX_PAGES = [
    ("수출환어음 매입", "/cms/fnm/sda_5/sda_54/sda_541/sda_5416/1186892_1366.html"),
    ("수출환어음 인수 후 매입", "/cms/fnm/sda_5/sda_54/sda_541/sda_5417/1186893_1368.html"),
    ("수출신용장 통지", "/cms/fnm/sda_5/sda_54/sda_541/sda_5413/1186882_1363.html"),
    ("무역금융", "/cms/fnm/sda_5/sda_55/sda_552/1186916_1387.html"),
    ("내국신용장", "/cms/fnm/sda_5/sda_55/sda_551/sda_5511/1186913_1384.html"),
    ("수입신용장 개설", "/cms/fnm/sda_5/sda_54/sda_544/sda_5444/1186909_1379.html"),
    ("수입화물선취보증서", "/cms/fnm/sda_5/sda_54/sda_544/sda_5446/1186911_1381.html"),
    ("수입물품대도", "/cms/fnm/sda_5/sda_54/sda_544/sda_5447/1186912_1382.html"),
]

# 항목별로 찾을 라벨(제목) 키워드 — 앞에 있는 키워드부터 차례로 찾는다(우선순위).
# 라벨에 키워드가 있고 제외어가 없어야 하며, 페이지 제목과 같은 라벨은 쓰지 않는다.
FIELD_LABELS = {
    "금리": (["대출금리", "보증요율", "보전금리", "금리"],
             ["연체", "우대", "감면", "적용방법", "고정금리", "변동금리"]),
    "한도": (["대출한도", "융자한도", "보증한도", "보증최고", "한도"], []),
    "기간": (["대출기간", "융자기간", "보증기간", "상환기간"], ["신청", "청약", "철회"]),
    "상환방식": (["상환 및 이자납입", "상환방법", "상환방식", "상환기간"], ["중도상환", "연체", "제한"]),
    "담보·보증": (["담보 또는 보증", "담보", "보증서", "보증비율"], ["보증료", "담보취득"]),
    "가입대상원문": (["대출대상", "가입대상", "신청자격", "융자대상", "지원대상", "대상기업", "보증 대상", "자격"], ["제외"]),
    "상품설명원문": (["상품특징", "개요", "상품개요", "상품설명", "상품소개"], []),
}
LOAN_FIELDS = ["금리", "한도", "기간", "상환방식", "담보·보증", "가입대상원문", "상품설명원문"]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class Fetcher:
    def __init__(self, cache_dir: str | None):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.last = 0.0
        self.n_requests = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get(self, path: str) -> str:
        url = urljoin(BASE, path)
        if self.cache_dir:
            cp = os.path.join(self.cache_dir, hashlib.md5(url.encode()).hexdigest() + ".html")
            if os.path.exists(cp):
                return open(cp, encoding="utf-8").read()
        wait = REQUEST_INTERVAL - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        r = self.session.get(url, timeout=30)
        self.last = time.time()
        self.n_requests += 1
        r.raise_for_status()
        html = r.content.decode("utf-8", errors="replace")
        if self.cache_dir:
            open(cp, "w", encoding="utf-8").write(html)
        return html


def meta_refresh(soup: BeautifulSoup) -> str | None:
    m = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
    if m and "url=" in m.get("content", "").lower():
        return m["content"].split("=", 1)[1].strip()
    return None


def list_products(fetch: Fetcher, list_name: str, list_url: str) -> list[dict]:
    """목록 페이지(페이지 이동 포함)에서 상품 상세 링크와 목록 설명을 모은다."""
    folder = list_url.rsplit("/", 1)[0]
    soup = BeautifulSoup(fetch.get(list_url), "html.parser")
    target = meta_refresh(soup)
    if target:   # 단일 상품 폴더
        return [{"list": list_name, "url": urljoin(BASE + list_url, target), "name": "", "list_desc": ""}]

    out, pages_done, queue = [], {list_url}, [soup]
    while queue:
        s = queue.pop(0)
        for a in s.find_all("a", href=True):
            href = a["href"]
            if re.search(re.escape(folder) + r"/\d+_\d+\.html$", href):
                row = a.find_parent("tr")
                desc = clean(row.get_text(" ", strip=True)) if row else ""
                name = clean(a.get_text(" ", strip=True))
                if desc.startswith(name):
                    desc = desc[len(name):].strip()
                out.append({"list": list_name, "url": urljoin(BASE, href), "name": name, "list_desc": desc})
            elif re.search(re.escape(folder) + r"/index,\d+,list\d+,\d+\.html$", href) and href not in pages_done:
                pages_done.add(href)
                queue.append(BeautifulSoup(fetch.get(href), "html.parser"))
    return out


def sections(soup: BeautifulSoup) -> dict[str, str]:
    """라벨 → 본문. (1) h2/h3/h4 제목 뒤 형제 요소들(다음 제목 전까지) (2) 표의 th → 같은 행 td."""
    out: dict[str, str] = {}
    body = soup.body or soup
    for h in body.find_all(["h2", "h3", "h4"]):
        label = clean(h.get_text(" ", strip=True))
        if not label or "produtTit" in " ".join(h.get("class", [])):
            continue
        parts = []
        for sib in h.find_next_siblings():
            if sib.name in ("h2", "h3", "h4"):
                break
            parts.append(clean(sib.get_text(" ", strip=True)))
        text = clean(" ".join(p for p in parts if p))
        if text and label not in out:
            out[label] = text
    for th in body.find_all("th"):
        td = th.find_next_sibling("td")
        label = clean(th.get_text(" ", strip=True))
        if td is not None and label and label not in out:
            out[label] = clean(td.get_text(" ", strip=True))
    # (3) 목록형 라벨: li.bult_tit 뒤 형제 li들 (다음 bult_tit 전까지) — 정책자금·보증서 페이지 구조
    for li in body.select("li.bult_tit"):
        label = clean(li.get_text(" ", strip=True))
        parts = []
        for sib in li.find_next_siblings("li"):
            if "bult_tit" in sib.get("class", []):
                break
            if "bult_11" in sib.get("class", []):   # bult_txt 안의 하위 항목이 따로 반복되므로 제외
                continue
            parts.append(clean(sib.get_text(" ", strip=True)))
        text = clean(" ".join(p for p in parts if p))
        if text and label and label not in out:
            out[label] = text
    # (4) 열 제목 표: 첫 행이 전부 th인 표 → 열 제목별로 "행이름: 값"을 이어 붙임
    for table in body.find_all("table"):
        trs = table.find_all("tr")
        if len(trs) < 2 or trs[0].find("td") is not None:
            continue
        if table.find(attrs={"rowspan": True}) or table.find(attrs={"colspan": True}):
            continue   # 병합 셀이 있으면 열 정렬을 확신할 수 없어 읽지 않음
        heads = [clean(c.get_text(" ", strip=True)) for c in trs[0].find_all(["th", "td"])]
        data = [[clean(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])] for tr in trs[1:]]
        for j, h in enumerate(heads[1:], start=1):
            vals = [f"{r[0]}: {r[j]}" for r in data if len(r) == len(heads) and r[j]]
            if h and vals and h not in out:
                out[h] = " / ".join(vals)
    return out


def pick(secs: dict[str, str], field: str, page_title: str = "") -> str | None:
    keys, excl = FIELD_LABELS[field]
    title_key = re.sub(r"\s+", "", page_title)
    for k in keys:                      # 우선순위 순서
        for label, text in secs.items():
            if re.sub(r"\s+", "", label) == title_key:
                continue
            if k in label and not any(x in label for x in excl):
                return text
    return None


DATE_RE = re.compile(r"(\d{4}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{1,2}\.?\s*(?:현재|기준))")


def classify_target(text: str) -> str:
    """가입대상 문구 기준. 법인·개인사업자가 둘 다 명시되면 둘다, 하나만이면 그쪽, 아니면 불명."""
    if not text or text == FAIL:
        return "불명"
    corp = bool(re.search(r"법인", text))
    sole = bool(re.search(r"개인사업자|개인기업|자영업|소상공인|개인 사업자", text))
    if corp and sole:
        return "둘다"
    if corp:
        return "법인"
    # "중소기업(개인사업자 포함)"처럼 기업 전반을 대상으로 하며 개인사업자를 덧붙인 문구는 애매하므로 불명
    if sole and not re.search(r"중소기업|중견기업|대기업|개인사업자\s*포함", text):
        return "개인사업자"
    return "불명"


def parse_detail(html: str, is_fx: bool) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.select_one("h2.produtTit3")
    name = clean(title_tag.get_text(" ", strip=True)) if title_tag else clean(soup.title.get_text() if soup.title else "")
    secs = sections(soup)
    row = {"상품명_상세": name, "라벨수": len(secs)}
    for f in LOAN_FIELDS:
        v = pick(secs, f, name)
        if v is None:
            v = "" if is_fx and f not in ("상품설명원문",) else FAIL
        row[f] = v
    if is_fx and row["상품설명원문"] == FAIL:
        # 외환 안내: 개요 라벨이 없으면 첫 업무 설명 절을 쓴다 (페이지 원문)
        first = next(iter(secs.values()), "")
        row["상품설명원문"] = first if first else FAIL
    rate = row["금리"]
    m = DATE_RE.search(rate) if rate and rate != FAIL else None
    row["금리기준일"] = clean(m.group(1)) if m else ""
    row["상품설명원문"] = row["상품설명원문"][:300]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="HTML 캐시 폴더 (선택)")
    ap.add_argument("--out", default="기업상품.csv")
    args = ap.parse_args()
    fetch = Fetcher(args.cache)
    today = dt.date.today().isoformat()

    # 1) 목록 → 상세 URL (중복 상품은 상품ID로 합치고 출처목록만 합침)
    items: dict[str, dict] = {}
    for list_name, list_url in LOAN_LISTS:
        prods = list_products(fetch, list_name, list_url)
        print(f"[목록] {list_name}: {len(prods)}개")
        for p in prods:
            pid = re.search(r"/(\d+)_\d+\.html$", p["url"]).group(1)
            it = items.setdefault(pid, {"상품ID": pid, "urls": [], "lists": [], "name": p["name"],
                                        "list_desc": p["list_desc"], "fx": False})
            if list_name not in it["lists"]:
                it["lists"].append(list_name)
            it["urls"].append(p["url"])
            if not it["name"]:
                it["name"] = p["name"]
            if not it["list_desc"]:
                it["list_desc"] = p["list_desc"]
    # 같은 상품이 목록마다 다른 페이지 번호로 올라온 경우(예: 운전자금·시설자금의 ESG Grow-Up)
    # 상품명(공백 제거)이 같으면 한 행으로 합친다. 먼저 나온 목록의 페이지를 대표로 쓴다.
    merged: dict[str, dict] = {}
    for pid, it in items.items():
        key = re.sub(r"\s+", "", it["name"]) or pid
        if key in merged:
            m = merged[key]
            m["lists"] += [x for x in it["lists"] if x not in m["lists"]]
            m["urls"] += it["urls"]
            print(f"[중복 합침] {it['name']}: {pid} → {m['상품ID']}")
        else:
            merged[key] = it
    items = {it["상품ID"]: it for it in merged.values()}
    for fx_name, fx_url in FX_PAGES:
        pid = re.search(r"/(\d+)_\d+\.html$", fx_url).group(1)
        items[pid] = {"상품ID": pid, "urls": [urljoin(BASE, fx_url)], "lists": ["외환안내"],
                      "name": fx_name, "list_desc": "", "fx": True}

    # 2) 상세 페이지
    rows = []
    for pid, it in items.items():
        url = it["urls"][0]
        d = parse_detail(fetch.get(url), it["fx"])
        if not it["fx"] and d["상품설명원문"] == FAIL and it["list_desc"]:
            d["상품설명원문"] = it["list_desc"][:300]   # 상세에 개요 라벨이 없으면 목록 페이지 설명(원문)
        if it["fx"]:
            kind = "외환"
        elif set(it["lists"]) & GUARANTEE_LISTS:
            kind = "보증"
        else:
            kind = "대출"
        rows.append({
            "상품ID": pid,
            "상품명": d["상품명_상세"] or it["name"],
            "출처목록": "|".join(it["lists"]),
            "유형": kind,
            "대상": classify_target(d["가입대상원문"]),
            "금리": d["금리"], "금리기준일": d["금리기준일"],
            "한도": d["한도"], "기간": d["기간"], "상환방식": d["상환방식"], "담보·보증": d["담보·보증"],
            "가입대상원문": d["가입대상원문"], "상품설명원문": d["상품설명원문"],
            "상세URL": url, "수집일": today,
        })
        print(f"[상세] {pid} {rows[-1]['상품명'][:30]} | 라벨 {d['라벨수']}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {args.out} ({len(out)}행), 실제 요청 {fetch.n_requests}회")


if __name__ == "__main__":
    main()
