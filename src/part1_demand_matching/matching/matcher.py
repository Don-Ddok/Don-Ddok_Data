# -*- coding: utf-8 -*-
"""상품 매칭 모델 3단계: 매칭 엔진 — 법인 × 월마다 추천 상품 최대 3개와 우선순위

입력 (읽기만 함):
  C:\\test\\분석결과\\matching\\trigger_stages.csv     지역 × 월 단계
  C:\\test\\분석결과\\matching\\persona_firm_month.csv 법인 × 월 세그먼트·페르소나·보유 태그
  C:\\test\\im뱅크상품\\기업상품_매핑.csv                상품표 (확정본)
규칙: match_rules.yaml (코드에는 규칙을 두지 않음)
출력 (C:\\test\\분석결과\\matching):
  recommend_firm_month.csv  법인 × 월 추천 (로컬 전용)
  recommend_summary_*.csv   집계표
실행: py -3.11 matcher.py

변경 이력
  - 2026-09-30: 우선순위 정렬을 사전식(잔액→최근변화)에서 점수화(가중합, match_rules.yaml priority.method)로
    바꿨다. 디벨롭 검토(outputs\\devreview\\bundle_h_ranking.py)에서 사전식 정렬은 반올림 동률 때문에
    전체로는 '최근 변화'가 넓게 작동하지만, 실제 상담에 쓰이는 상위권(정점·유지 상위 20위)에는 전혀
    반영되지 않음(그 기준을 넣든 빼든 순위 100% 동일)을 확인했다. 이전 방식은 priority.method:
    lexicographic으로 되돌릴 수 있다. 사용자 승인(2026-09-30).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 경로 설정 ----------------
BASE = r"C:\test\분석결과\matching"
TRIGGER_FILE = os.path.join(BASE, "trigger_stages.csv")
PERSONA_FILE = os.path.join(BASE, "persona_firm_month.csv")
PRODUCT_FILE = r"C:\test\im뱅크상품\기업상품_매핑.csv"
RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "match_rules.yaml")
OUT_FILE = os.path.join(BASE, "recommend_firm_month.csv")
KEY_COLS = ["단계", "페르소나", "세그먼트", "보유_A", "보유_B", "보유_C", "업종_대분류", "지역"]
N_FIELDS = ["상품명", "모델유형", "법인추천가능", "추가요건", "제안방식", "이용경로"]


def parse_filter(expr: str):
    """'컬럼=값' 또는 '컬럼 in (값,값)' → (컬럼, 허용값 집합). 빈 문자열이면 None."""
    expr = (expr or "").strip()
    if not expr:
        return None
    m = re.fullmatch(r"(\S+)\s+in\s+\(([^)]*)\)", expr)
    if m:
        return m.group(1), {v.strip() for v in m.group(2).split(",")}
    m = re.fullmatch(r"([^=\s]+)\s*=\s*(.+)", expr)
    if m:
        return m.group(1), {m.group(2).strip()}
    raise ValueError(f"필터 문법 오류: {expr}")


def filter_pass(expr: str, firm: dict) -> bool:
    f = parse_filter(expr)
    if f is None:
        return True
    col, allowed = f
    if col not in firm:              # 법인 데이터에 없는 컬럼 → 통과시키지 않음
        return False
    return str(firm[col]) in allowed


class Matcher:
    def __init__(self, rules: dict, products: pd.DataFrame):
        self.r = rules
        p = products[~products["모델유형"].isin(rules["exclude_types"])]
        p = p[p["법인추천가능"].isin(rules["recommend_order"])].copy()
        order = {v: i for i, v in enumerate(rules["recommend_order"])}
        p["_rec"] = p["법인추천가능"].map(order)
        kw = rules.get("demote_note_keyword")
        if kw:   # 비고에 키워드가 있으면 Y라도 확인필요와 같은 정렬 순위 (표시값은 그대로)
            demote = p["비고"].str.contains(kw, regex=False) & (p["법인추천가능"] == "Y")
            p.loc[demote, "_rec"] = order.get("확인필요", len(order))
        p["_id"] = p["상품ID"].astype(int)
        self.p = p
        self.stage_row_types = {st: {t for cell in cells.values() for t in cell.get("types", [])}
                                for st, cells in rules["stage_persona"].items()}

    def candidates(self, mtype: str, stage: str) -> pd.DataFrame:
        opt = (self.r.get("stage_options") or {}).get(stage, {})
        use_sub = opt.get("match_sub_type", self.r.get("match_sub_type"))
        main = self.p["모델유형"] == mtype
        hit = main | (self.p["보조유형"] == mtype) if use_sub else main
        c = self.p[hit].assign(_sub=(~main[hit]).astype(int) if self.r.get("sub_type_after_main") else 0)
        if c.empty:
            return c
        # 유형별 기본순위: groups를 위에서부터, "*"는 그 외(상품ID 순). 설정이 없으면 상품ID 순
        groups = (self.r.get("product_rank") or {}).get(mtype, {}).get("groups", ["*"])
        star = groups.index("*") if "*" in groups else len(groups)

        def rank(row):
            for gi, g in enumerate(groups):
                if g != "*" and row["상품명"] in g:
                    return (gi, g.index(row["상품명"]))
            return (star, row["_id"])
        c["_rank"] = c.apply(rank, axis=1)
        return c.sort_values(["_sub", "_rec", "_rank"])

    def type_list(self, key: dict) -> tuple[list, str | None]:
        """(모델유형 우선순위, 칸 제안방식). 보조 태그 유형은 뒤에 붙인다."""
        st = key["단계"]
        cells = self.r["stage_persona"][st]
        if st == self.r["general_stage"]:
            cell = cells[self.r["segments"]["none"]]
        else:
            cell = cells[key["페르소나"][:1]]
        method = cell.get("method")
        if method in self.r.get("hold_methods", []):
            return [], method
        types = list(cell.get("types", []))
        scope = self.r["subtag_scope"]
        scope = scope.get(st, "cell") if isinstance(scope, dict) else scope
        allowed = self.stage_row_types[st] if scope in ("row", "stage_row") else set(types)
        for s in self.r["subtag_types"]:
            if key.get(s["tag"]) == "Y" and s["type"] in allowed and s["type"] not in types:
                types.append(s["type"])
        return types, method

    def held(self, prod: pd.Series, key: dict) -> bool:
        for rule in self.r["held_adjust"]["rules"]:
            if key.get(rule["tag"]) != "Y":
                continue
            if prod["상품명"] in rule.get("product_names", []):
                return True
            if {prod["모델유형"], prod["보조유형"]} & set(rule.get("types", [])):
                return True
        return False

    def usage_path(self, prod: pd.Series, key: dict) -> str:
        cfg = self.r["usage_path"]
        if prod["모델유형"] != cfg["for_type"]:
            return ""
        if prod["상품명"] in cfg.get("by_product", {}):
            return cfg["by_product"][prod["상품명"]]
        return cfg.get("default_by_region", {}).get(key["지역"], "")

    def recommend(self, key: dict) -> list[dict]:
        types, cell_method = self.type_list(key)
        base_method = cell_method or self.r["stage_methods"][key["단계"]]
        n_max = self.r["max_recommendations"]
        cr = self.r["count_rule"]
        st = key["단계"]
        # 유형별 적격 후보 (필터 통과). 칸 유형 수 = 적격 후보가 있는 유형 수
        elig = {}
        for t in types:
            c = self.candidates(t, st)
            ok = [prod for _, prod in c.iterrows() if filter_pass(prod["데이터필터"], key)]
            if ok:
                elig[t] = ok
        n_types = len(elig)
        per_type = cr["single_type_max"] if n_types == 1 else cr["per_type_multi"]
        limit = min(n_max, per_type * n_types)

        out, seen, used_model = [], set(), set()
        for t, prods in elig.items():
            taken = 0
            for prod in prods:
                if len(out) >= limit or taken >= per_type:
                    break
                if prod["상품ID"] in seen:
                    continue
                # 칸 유형이 여러 개면, 이미 뽑힌 모델유형의 상품은 다른 유형 칸에서 다시 쓰지 않음
                if n_types > 1 and prod["모델유형"] in used_model:
                    continue
                seen.add(prod["상품ID"])
                used_model.add(prod["모델유형"])
                taken += 1
                method = self.r["held_adjust"]["method"] if self.held(prod, key) else base_method
                out.append({"상품명": prod["상품명"], "모델유형": prod["모델유형"], "법인추천가능": prod["법인추천가능"],
                            "추가요건": prod["추가요건"], "제안방식": method,
                            "이용경로": self.usage_path(prod, key), "_type_slot": t, "_id": prod["상품ID"],
                            "_filter": prod["데이터필터"], "_n_types": n_types})
        return out


def main():
    rules = yaml.safe_load(open(RULES_FILE, encoding="utf-8"))
    products = pd.read_csv(PRODUCT_FILE, dtype=str).fillna("")
    trig = pd.read_csv(TRIGGER_FILE, dtype={"k": "Int64"})
    firm = pd.read_csv(PERSONA_FILE, dtype={"법인ID": str}).fillna({"페르소나": "", "보유_A": "", "보유_B": "",
                                                                    "보유_C": "", "보유_D": ""})
    exposed = firm["세그먼트"].isin(rules["segments"]["exposed"])

    # 1. 단계 부여
    firm = firm.merge(trig[["지역", "연월", "단계", "k"]], on=["지역", "연월"], how="left")
    firm.loc[~exposed, "단계"] = rules["general_stage"]
    firm.loc[~exposed, "k"] = pd.NA
    assert firm["단계"].notna().all(), "단계가 없는 행이 있음"

    # 2~6. 조합별 추천 (같은 조합은 추천이 같으므로 한 번만 계산)
    m = Matcher(rules, products)
    combos = firm[KEY_COLS].fillna("").drop_duplicates()
    rec_rows, raw = [], []
    for key in combos.to_dict("records"):
        recs = m.recommend(key)
        row = dict(key)
        for i in range(rules["max_recommendations"]):
            for f in N_FIELDS:
                row[f"추천{i + 1}_{f}"] = recs[i][f] if i < len(recs) else ""
        rec_rows.append(row)
        raw += [{**key, "순서": i + 1, **r} for i, r in enumerate(recs)]
    combo_rec = pd.DataFrame(rec_rows)
    raw = pd.DataFrame(raw)
    print(f"[조합] {len(combos):,}개, 조합별 추천 {len(raw):,}건")

    # 9. 점검 (조합 단위 — 법인×월 결과는 조합 결과를 그대로 붙인 것)
    none_seg = rules["segments"]["none"]
    bad_types = {"수출금융", "수입금융", "수출입운전자금"}
    assert raw[(raw["세그먼트"] == none_seg) & raw["모델유형"].isin(bad_types)].empty, "비노출에 수출입 유형 추천"
    assert all(filter_pass(r["_filter"], r) for r in raw.to_dict("records")), "필터 불통과 상품 추천"
    assert not raw["모델유형"].isin(rules["exclude_types"]).any(), "제외·보증기관안내 추천"
    assert not (raw["법인추천가능"] == "N").any(), "법인추천가능 N 추천"
    region_f = raw[raw["_filter"].str.startswith("지역")]
    assert all(filter_pass(r["_filter"], r) for r in region_f.to_dict("records")), "지역 필터 불통과 상품 추천"
    print(f"[점검] 5개 assert 통과 (지역 필터 상품 추천 {len(region_f)}건 모두 지역 일치)")
    g = raw.groupby(KEY_COLS)
    dup = g["모델유형"].apply(lambda s: s.duplicated().any())
    single = g["_n_types"].first() == 1
    assert not (dup & ~single).any(), "칸 유형이 2개 이상인데 같은 모델유형이 2개 이상 추천됨"
    assert (g.size() <= g["_n_types"].first().clip(upper=rules["max_recommendations"])
            .where(~single, rules["count_rule"]["single_type_max"])).all(), "추천 개수 규칙 위반"
    print(f"[점검] 추천 개수 규칙 assert 통과 | 같은 모델유형 2개 이상인 조합: {int(dup.sum())} / {len(dup)} "
          f"(모두 칸 유형 1개인 조합)")

    # 7. 법인 × 월 결과
    out = firm.merge(combo_rec, on=KEY_COLS, how="left")
    tb = rules["timing_basis"]
    out["근거_타이밍"] = out["단계"].map(tb)
    out["근거_상품"] = rules["product_basis"]
    # 제안메시지: (단계, 페르소나, 세그먼트) 덮어쓰기 → 없으면 (단계, 페르소나) 칸 (비노출은 일반 칸)
    mt = rules.get("message_templates") or {}
    seg_ov = mt.get("segment_overrides") or {}
    none_key = rules["segments"]["none"]

    def message(st, p, seg):
        key = none_key if st == rules["general_stage"] else str(p)[:1]
        ov = ((seg_ov.get(st) or {}).get(key) or {}).get(seg)
        return ov if ov is not None else (mt.get(st) or {}).get(key, "")
    out["제안메시지"] = [message(st, p, seg) for st, p, seg in zip(out["단계"], out["페르소나"], out["세그먼트"])]

    # 점검: 세그먼트 방향과 맞지 않는 단어 (중립어는 빼고 봄)
    mc = rules.get("message_check") or {}
    neutral = mc.get("neutral_words", [])
    for seg, bad in mc.items():
        if seg == "neutral_words":
            continue
        msgs = out.loc[out["세그먼트"] == seg, "제안메시지"].drop_duplicates()
        stripped = msgs.apply(lambda m: re.sub("|".join(map(re.escape, neutral)), "", m) if neutral else m)
        wrong = msgs[stripped.str.contains(bad, regex=False)]
        assert wrong.empty, f"{seg} 제안메시지에 '{bad}' 포함: {wrong.tolist()}"
    print("[점검] 제안메시지 세그먼트 방향 assert 통과")

    # 8. 우선순위 (월마다)
    pr = rules["priority"]
    n = pr["demand_change_months"]
    out["_m"] = (out["연월"] // 100) * 12 + out["연월"] % 100
    prev = out[["법인ID", "_m", "요구불잔액"]].rename(columns={"요구불잔액": "_prev"})
    prev["_m"] = prev["_m"] + n
    out = out.merge(prev, on=["법인ID", "_m"], how="left")
    out["_chg"] = out["요구불잔액"] - out["_prev"]
    out["_st"] = out["단계"].map({s: i for i, s in enumerate(pr["stage_order"])})
    out["_seg"] = np.where(out["세그먼트"].isin(rules["segments"]["exposed"]), 0, 1)

    if pr.get("method") == "score":
        # 그룹(연월×단계×노출여부) 안에서 백분위 순위의 가중합. 큰 값일수록 백분위가 1에 가깝도록
        # rank(ascending=True)를 쓰고, '최근 변화'는 감소가 클수록(음수가 작을수록) 유리하도록 -_chg를 랭크한다.
        w, miss = pr["weights"], pr["missing_momentum_percentile"]
        grp = out.groupby(["연월", "_st", "_seg"])
        out["_size_pct"] = grp["요구불잔액"].transform(lambda s: s.rank(pct=True, ascending=True))
        out["_mom_pct"] = grp["_chg"].transform(lambda s: (-s).rank(pct=True, ascending=True)).fillna(miss)
        out["_score"] = w["size"] * out["_size_pct"] + w["momentum"] * out["_mom_pct"]
        out = out.sort_values(["연월", "_st", "_seg", "_score"], ascending=[True, True, True, False], kind="mergesort")
    else:
        out = out.sort_values(["연월", "_st", "_seg", "요구불잔액", "_chg"], ascending=[True, True, True, False, True],
                              na_position="last")
    out["우선순위순번"] = out.groupby("연월").cumcount() + 1

    cols = ["법인ID", "연월", "지역", "업종_대분류", "세그먼트", "페르소나", "단계", "k"]
    for i in range(rules["max_recommendations"]):
        cols += [f"추천{i + 1}_{f}" for f in N_FIELDS]
    cols += ["제안메시지", "근거_타이밍", "근거_상품", "우선순위순번"]
    out = out.sort_values(["연월", "우선순위순번"])
    out[cols].to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_FILE} ({len(out):,}행)")
    return out, raw, rules


if __name__ == "__main__":
    main()
