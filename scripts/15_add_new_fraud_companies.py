"""
[표본 추가] 새로 확인한 분식기업 2개(일양약품, 에스케이엔펄스)의 재무데이터를 조회하여
           기존 beneish_scores.csv에 추가합니다.

실행하면 -> beneish_scores.csv가 갱신됩니다 (기존 11개 + 신규 최대 2개).
이후 14_refreeze_training_data.py를 다시 실행하면 training_data_final.csv에 자동 반영됩니다.
"""

import requests
import pandas as pd
import time

API_KEY = "여기에_본인_DART_API_키를_입력하세요"
BASE_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

NEW_FRAUD_COMPANIES = [
    {"corp_name": "일양약품", "corp_code": "00146454", "year": 2023},
    {"corp_name": "에스케이엔펄스", "corp_code": "00307189", "year": 2020},
]

ACCOUNT_KEYWORDS = {
    "sales": ["매출액", "수익(매출액)", "영업수익"],
    "cogs": ["매출원가"],
    "receivables": ["매출채권", "외상매출금"],
    "current_assets": ["유동자산"],
    "ppe": ["유형자산"],
    "total_assets": ["자산총계"],
    "sga": ["판매비와관리비", "판매비와 관리비"],
    "total_liabilities": ["부채총계"],
    "current_liabilities": ["유동부채"],
    "cash": ["현금및현금성자산"],
    "depreciation": ["감가상각비"],
}


def find_account_value(items, keywords, amount_key):
    for kw in keywords:
        for it in items:
            if kw in it.get("account_nm", ""):
                val = it.get(amount_key, "")
                val = val.replace(",", "") if val else ""
                if val not in ("", None):
                    try:
                        return float(val)
                    except ValueError:
                        continue
    return None


def fetch_financial_year(corp_code, year, fs_div="CFS"):
    params = {
        "crtfc_key": API_KEY, "corp_code": corp_code, "bsns_year": str(year),
        "reprt_code": "11011", "fs_div": fs_div,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    data = resp.json()
    if data.get("status") != "000":
        return None, None
    items = data.get("list", [])
    t_data, t1_data = {}, {}
    for key, kws in ACCOUNT_KEYWORDS.items():
        t_data[key] = find_account_value(items, kws, "thstrm_amount")
        t1_data[key] = find_account_value(items, kws, "frmtrm_amount")
    return t_data, t1_data


def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_ratios(t, t1):
    r1 = safe_div(t["receivables"], t["sales"])
    r2 = safe_div(t1["receivables"], t1["sales"])
    dsri = safe_div(r1, r2) or 1.0

    if t["cogs"] is not None and t1["cogs"] is not None and t["sales"] and t1["sales"]:
        gm_t = safe_div(t["sales"] - t["cogs"], t["sales"])
        gm_t1 = safe_div(t1["sales"] - t1["cogs"], t1["sales"])
        gmi = safe_div(gm_t1, gm_t) or 1.0
    else:
        gmi = 1.0

    if None not in (t["current_assets"], t["ppe"], t["total_assets"],
                     t1["current_assets"], t1["ppe"], t1["total_assets"]):
        aqi_t = safe_div(t["total_assets"] - t["current_assets"] - t["ppe"], t["total_assets"])
        aqi_t1 = safe_div(t1["total_assets"] - t1["current_assets"] - t1["ppe"], t1["total_assets"])
        aqi = safe_div(aqi_t, aqi_t1) or 1.0
    else:
        aqi = 1.0

    sgi = safe_div(t["sales"], t1["sales"]) or 1.0

    depi = None
    if t["depreciation"] and t1["depreciation"] and t["ppe"] and t1["ppe"]:
        dep_rate_t = safe_div(t["depreciation"], t["depreciation"] + t["ppe"])
        dep_rate_t1 = safe_div(t1["depreciation"], t1["depreciation"] + t1["ppe"])
        depi = safe_div(dep_rate_t1, dep_rate_t)
    depi = depi or 1.0

    s1 = safe_div(t["sga"], t["sales"])
    s2 = safe_div(t1["sga"], t1["sales"])
    sgai = safe_div(s1, s2) or 1.0

    l1 = safe_div(t["total_liabilities"], t["total_assets"])
    l2 = safe_div(t1["total_liabilities"], t1["total_assets"])
    lvgi = safe_div(l1, l2) or 1.0

    tata = None
    if None not in (t["current_assets"], t["cash"], t["current_liabilities"], t["total_assets"],
                     t1["current_assets"], t1["cash"], t1["current_liabilities"]) and t["total_assets"]:
        dep_for_tata = t["depreciation"] if t["depreciation"] else 0
        wc_t = (t["current_assets"] - t["cash"]) - t["current_liabilities"]
        wc_t1 = (t1["current_assets"] - t1["cash"]) - t1["current_liabilities"]
        tata = safe_div(wc_t - wc_t1 - dep_for_tata, t["total_assets"])
    tata = tata if tata is not None else 0.0

    return {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
            "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata}


results = []
for comp in NEW_FRAUD_COMPANIES:
    corp_code, corp_name, year = comp["corp_code"], comp["corp_name"], comp["year"]
    print(f"조회 중: {corp_name} ({year}년)")
    t, t1 = fetch_financial_year(corp_code, year, fs_div="CFS")
    if t is None or all(v is None for v in t.values()):
        t, t1 = fetch_financial_year(corp_code, year, fs_div="OFS")
    if t is None:
        print("  -> 데이터 조회 실패")
        time.sleep(0.3)
        continue
    ratios = compute_ratios(t, t1)
    row = {"corp_name": corp_name, "corp_code": corp_code, "year": year,
           "m_score": None, "flag_high_risk": None}
    row.update(ratios)
    results.append(row)
    print(f"  -> 계산 완료: {ratios}")
    time.sleep(0.3)

new_df = pd.DataFrame(results)

# 기존 beneish_scores.csv와 합치기 (중복 제거)
existing = pd.read_csv("beneish_scores.csv")
combined = pd.concat([existing, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=["corp_code"])

combined.to_csv("beneish_scores.csv", index=False, encoding="utf-8-sig")
print(f"\n갱신 완료. 분식기업 총 {len(combined)}개 -> beneish_scores.csv 저장")
print("이제 14_refreeze_training_data.py를 다시 실행하면 training_data_final.csv에 반영됩니다.")
