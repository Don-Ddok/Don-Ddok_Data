"""iM뱅크 공개 금융상품 페이지 크롤러

결과 파일 (실행 폴더에 생성):
  imbank_product_pages.csv  방문한 페이지별 제목/본문 요약
  imbank_product_tables.csv 페이지에 표시된 표의 행 데이터
  imbank_product_links.csv  상품 상세 페이지로 연결되는 링크

실행 전:
  pip install selenium beautifulsoup4 pandas

Selenium Manager가 ChromeDriver를 자동으로 준비합니다. Chrome이 설치되어 있어야 합니다.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

BASE = "https://www.imbank.co.kr"
HUB = f"{BASE}/com_ebz_sm_fpm_sub_main.act"

# 상품몰에서 공개적으로 조회할 수 있는 주요 상품목록 페이지들.
# 메인 상품몰에서 발견되는 예금/대출/카드/펀드/보험/신탁/퇴직연금 링크도 함께 따라갑니다.
SEED_URLS = [
    HUB,
    f"{BASE}/fnp_ebz_21000_depo.act",                 # 예금 인터넷 신규상품
    f"{BASE}/fnp_ebz_31010_depo.act?deopDv=1",        # 목돈만들기(적금 등)
    f"{BASE}/fnp_ebz_31010_depo.act?deopDv=2",        # 목돈굴리기(예금 등)
    f"{BASE}/fnl_ebz_43010_loan.act",                 # 대출 인터넷 신규상품
    f"{BASE}/fnc_ebz_11010_card.act",                 # 개인 신용카드 목록
    f"{BASE}/fnc_ebz_11011_card.act",                 # 개인 체크카드 목록
    f"{BASE}/fnf_ebz_38010_fund.act",                 # 펀드 기준가/수익률 목록
    f"{BASE}/fnf_ebz_31030_fund.act",                 # 국내 펀드 카테고리(채권형 등)
]

CATEGORY_WORDS = {
    "예금": ["예금", "적금", "depo", "deposit"],
    "대출": ["대출", "loan"],
    "카드": ["카드", "card"],
    "펀드": ["펀드", "fund"],
    "보험": ["보험", "방카", "insurance"],
    "신탁": ["신탁", "trust"],
    "퇴직연금": ["퇴직연금", "연금", "pension", "irp"],
    "외환": ["외환", "외화예금", "환전", "송금", "exchange", "forex"],
}

# 상품과 무관한 공통 메뉴/로그인/업무 페이지는 건너뜁니다.
SKIP_WORDS = [
    "login", "logout", "auth", "cert", "agree", "join", "mypage",
    "transfer", "inquiry", "search", "popup", "event", "notice",
    "customer", "counsel", "faq", "branch", "download", "terms",
    "privacy", "error", "javascript:", "mailto:", "tel:", "#",
]
NAV_LABELS = {
    "자세히보기", "더보기", "이전", "다음", "검색", "검색하기", "초기화",
    "로그인", "전체메뉴", "금융상품몰", "상품검색", "관심상품", "최근 본 상품",
}
MAX_PAGES = 250       # 안전을 위한 최대 방문 페이지 수
MAX_DEPTH = 3         # 상품몰에서 상세 페이지까지의 링크 단계
PAGE_WAIT_SECONDS = 1.2


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def categorize(url: str, text: str = "") -> str:
    haystack = f"{url} {text}".lower()
    for category, words in CATEGORY_WORDS.items():
        if any(word.lower() in haystack for word in words):
            return category
    return "금융상품몰"


def normalize_url(href: str, current_url: str) -> str | None:
    if not href:
        return None
    url = urldefrag(urljoin(current_url, href))[0]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() not in {
        "www.imbank.co.kr", "imbank.co.kr"
    }:
        return None
    # 같은 페이지로 연결되는 과도한 tracking 파라미터는 그대로 두되 URL fragment는 제거
    return url


def should_skip(url: str) -> bool:
    lower = url.lower()
    return any(word in lower for word in SKIP_WORDS)


def relevant_link(label: str, href: str, parent_category: str) -> bool:
    label = clean(label)
    if not label or label in NAV_LABELS or len(label) < 2 or should_skip(href):
        return False
    # 카테고리 키워드가 URL/링크명에 있거나, 상품목록 페이지 안에서 상품명 링크로 보이는 경우
    if categorize(href, label) != "금융상품몰":
        return True
    product_words = ["상품", "예금", "적금", "대출", "카드", "펀드", "보험", "신탁", "연금"]
    if any(word in label for word in product_words):
        return True
    # 개별 상품명은 일반적인 단어일 수 있으므로, 상품 카테고리 페이지 안의 링크도 수집
    return parent_category != "금융상품몰" and len(label) >= 3


def extract_tables(soup: BeautifulSoup, page_url: str, title: str, category: str):
    records = []
    for table_no, table in enumerate(soup.find_all("table"), start=1):
        rows = []
        for tr in table.find_all("tr"):
            cells = [clean(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(cells)
        if not rows:
            continue
        # 페이지 전체의 메뉴/빈 표는 제외하고 실제 데이터가 있는 표만 저장
        if len(rows) == 1 and len(rows[0]) <= 1:
            continue
        records.append({
            "category": category,
            "page_title": title,
            "page_url": page_url,
            "table_no": table_no,
            "rows_json": json.dumps(rows, ensure_ascii=False),
        })
    return records


def make_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1100")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.page_load_strategy = "eager"
    return webdriver.Chrome(options=options)


def main():
    queue = deque((url, 0, categorize(url)) for url in SEED_URLS)
    seen = set()
    page_rows = []
    table_rows = []
    product_links = []

    driver = make_driver()
    try:
        while queue and len(seen) < MAX_PAGES:
            url, depth, inherited_category = queue.popleft()
            if url in seen or should_skip(url):
                continue
            seen.add(url)

            try:
                driver.get(url)
                try:
                    WebDriverWait(driver, 20).until(
                        lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
                    )
                except TimeoutException:
                    pass
                time.sleep(PAGE_WAIT_SECONDS)  # 동적 상품 목록이 표시될 시간을 조금 줌
                current_url = driver.current_url
                soup = BeautifulSoup(driver.page_source, "html.parser")
            except (TimeoutException, WebDriverException) as exc:
                print(f"[건너뜀] {url} ({type(exc).__name__})")
                continue

            title = clean(soup.title.get_text(" ", strip=True)) if soup.title else ""
            headings = " ".join(clean(x.get_text(" ", strip=True)) for x in soup.select("h1, h2, h3"))
            page_text = clean(soup.get_text(" ", strip=True))
            category = categorize(current_url, f"{title} {headings}")
            if category == "금융상품몰":
                category = inherited_category

            # 너무 긴 페이지 본문 대신 앞부분만 저장. 원문 링크도 함께 기록합니다.
            page_rows.append({
                "category": category,
                "title": title,
                "url": current_url,
                "headings": headings[:1000],
                "text_preview": page_text[:5000],
            })
            table_rows.extend(extract_tables(soup, current_url, title, category))

            links_found = 0
            if depth < MAX_DEPTH:
                for a in soup.select("a[href]"):
                    href = normalize_url(a.get("href", ""), current_url)
                    label = clean(a.get_text(" ", strip=True))
                    if not href or href in seen or not relevant_link(label, href, category):
                        continue
                    target_category = categorize(href, label)
                    if target_category == "금융상품몰":
                        target_category = category
                    product_links.append({
                        "category": target_category,
                        "product_or_link_name": label,
                        "url": href,
                        "source_page": current_url,
                    })
                    queue.append((href, depth + 1, target_category))
                    links_found += 1

            print(f"[{len(seen):03d}] {category} | {title[:50]} | 표 {len(table_rows)}행 누적 | 링크 {links_found}개")

    finally:
        driver.quit()

    pd.DataFrame(page_rows).drop_duplicates(subset=["url"]).to_csv(
        "imbank_product_pages.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(table_rows).to_csv(
        "imbank_product_tables.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(product_links).drop_duplicates(subset=["url", "source_page"]).to_csv(
        "imbank_product_links.csv", index=False, encoding="utf-8-sig"
    )

    print("\n완료")
    print(f"방문 페이지: {len(page_rows)}개")
    print(f"표 데이터: {len(table_rows)}개")
    print(f"상품/관련 링크: {len(product_links)}개")
    print("생성 파일: imbank_product_pages.csv, imbank_product_tables.csv, imbank_product_links.csv")
    print("※ 상품 목록이 빈 표로 저장되면 페이지가 로그인/검색 또는 별도 조회를 요구하는지 확인하세요.")


if __name__ == "__main__":
    main()
