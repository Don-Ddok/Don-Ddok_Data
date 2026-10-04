"""기업상품.csv → 기업상품_매핑.csv (모델유형·매핑근거·법인추천가능·비고 추가)

규칙은 product_rules.yaml에 있다. 기업상품.csv는 읽기만 한다.
실행: python map_products.py [--rules product_rules.yaml] [--src 기업상품.csv] [--out 기업상품_매핑.csv]
"""
from __future__ import annotations

import argparse
import re

import pandas as pd
import yaml


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def name_match(name: str, rule: dict) -> bool:
    n, key = norm(name), norm(rule["name"])
    return n == key if rule.get("match", "exact") == "exact" else key in n


def map_type(row: pd.Series, rules: dict) -> tuple[str, str]:
    for r in rules.get("name_exceptions", []):
        if name_match(row["상품명"], r):
            return r["type"], f"상품명 예외({r['name']})"
    lists = str(row["출처목록"]).split("|")
    for r in rules.get("list_rules", []):
        if r["list"] in lists:
            return r["type"], f"목록({r['list']})"
    return "", "규칙 없음"


def recommend(row: pd.Series, rules: dict) -> str:
    for r in rules.get("recommend_rules", []):
        if all(str(row.get(k, "")) == str(v) for k, v in r["when"].items()):
            return r["value"]
    return "확인필요"


def notes(row: pd.Series, rules: dict) -> str:
    cfg = rules.get("notes", {})
    out = []
    m = re.search(r"(\d{4})", str(row.get("금리기준일", "")))
    if m and int(m.group(1)) < int(cfg.get("rate_date_before_year", 0)):
        out.append(cfg.get("rate_date_note", ""))
    for r in cfg.get("by_name", []):
        if name_match(row["상품명"], r):
            out.append(r["note"])
    return "; ".join(x for x in out if x)


def override_hit(row: pd.Series, m: dict) -> bool:
    if "list" in m:
        return m["list"] in str(row["출처목록"]).split("|")
    return name_match(row["상품명"], m)


def sub_type(row: pd.Series, rules: dict) -> str:
    lists = set(str(row["출처목록"]).split("|"))
    for r in rules.get("sub_type_rules", []):
        if set(r["lists_all"]) <= lists:
            return r["sub_type"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="product_rules.yaml")
    ap.add_argument("--src", default="기업상품.csv")
    ap.add_argument("--out", default="기업상품_매핑.csv")
    args = ap.parse_args()

    rules = yaml.safe_load(open(args.rules, encoding="utf-8"))
    df = pd.read_csv(args.src, dtype=str).fillna("")
    mapped = df.apply(lambda r: map_type(r, rules), axis=1, result_type="expand")
    df["모델유형"], df["매핑근거"] = mapped[0], mapped[1]
    df["보조유형"] = df.apply(lambda r: sub_type(r, rules), axis=1)
    df["법인추천가능"] = ""
    df["추가요건"] = ""
    df["데이터필터"] = ""
    df["필터근거"] = ""
    df["비고"] = df.apply(lambda r: notes(r, rules), axis=1)

    # 개별 조정: 모델유형·매핑근거·법인추천가능은 덮어쓰고, 비고는 이어 붙인다
    for o in rules.get("overrides", []):
        hit = df.apply(lambda r: override_hit(r, o["match"]), axis=1)
        for col, val in o["set"].items():
            if col == "비고":
                df.loc[hit, "비고"] = df.loc[hit, "비고"].apply(lambda x: "; ".join(p for p in [x, val] if p))
            else:
                df.loc[hit, col] = val
    todo = df["법인추천가능"] == ""
    df.loc[todo, "법인추천가능"] = df[todo].apply(lambda r: recommend(r, rules), axis=1)
    # 모델유형별 데이터필터 (상품별 필터가 없을 때만)
    for mtype, flt in (rules.get("type_filters") or {}).items():
        hit = (df["모델유형"] == mtype) & (df["데이터필터"] == "")
        df.loc[hit, "데이터필터"] = flt
        df.loc[hit, "필터근거"] = rules.get("type_filter_basis", "")

    tail = ["법인추천가능", "추가요건", "데이터필터", "필터근거", "비고"]
    cols = [c for c in df.columns if c not in ["보조유형"] + tail]
    cols.insert(cols.index("모델유형") + 1, "보조유형")
    df = df[cols + tail]
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"저장: {args.out} ({len(df)}행)")
    tab = pd.crosstab(df["모델유형"].replace("", "(없음)"), df["법인추천가능"], margins=True, margins_name="합계")
    print("\n[모델유형 × 법인추천가능]")
    print(tab.to_string())
    sub = df[df["보조유형"] != ""]
    print("\n[보조유형]", pd.crosstab(sub["보조유형"], sub["법인추천가능"]).to_string() if len(sub) else "없음")
    none = df[df["모델유형"] == ""]
    print("\n[모델유형 없음]", "없음" if none.empty else none[["상품명", "출처목록"]].values.tolist())
    for t in ["한도대출", "매출채권유동화", "수출금융"]:
        print(f"\n[{t}]", df.loc[df["모델유형"] == t, "상품명"].tolist())
    print("\n[비고]", df.loc[df["비고"] != "", ["상품명", "비고"]].values.tolist())


if __name__ == "__main__":
    main()
