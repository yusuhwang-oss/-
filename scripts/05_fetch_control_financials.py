"""
[5단계] 정상기업(비교집단)의 재무제표를 조회하고 Beneish 8개 지표를 계산.
- 03번 스크립트와 계산 로직은 동일하고, 대상만 control_labels.csv로 바뀐 버전입니다.

실행 전 준비물: control_labels.csv (4단계 결과물)
실행하면 -> control_scores.csv 파일이 생성됩니다.
주의: 정상기업 33개를 조회하는 거라 03번보다 시간이 좀 더 걸립니다 (API 호출 간 0.3초씩 대기).
"""

import requests
import pandas as pd
import time

API_KEY = "여기에_본인_DART_API_키를_입력하세요"
BASE_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

ACCOUNT_KEYWORDS = {
    "sales": ["매출액", "수익(매출액)", "영업수익"],
    "cogs": ["매출원가"],
    "receivables": ["매출채권", "외상매출금"],
    "current_assets": ["유동자산"],
    "ppe": ["유형자산"],
    "total_assets": ["자산총계"],
    "sga": ["판매비와관리비", "판매비와 관리비"],
    "total_liabilities": ["부채총계"],
    "total_equity": ["자본총계"],
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


def fetch_financial_year(corp_code, year, api_key, fs_div="CFS", reprt_code="11011"):
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(int(year)),
        "reprt_code": reprt_code,
        "fs_div": fs_div,
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


def compute_beneish_m(t, t1):
    missing_fields = []
    try:
        dsri = (t["receivables"] / t["sales"]) / (t1["receivables"] / t1["sales"])
    except (TypeError, ZeroDivisionError):
        missing_fields.append("DSRI")
        dsri = None
    try:
        gm_t = (t["sales"] - t["cogs"]) / t["sales"]
        gm_t1 = (t1["sales"] - t1["cogs"]) / t1["sales"]
        gmi = gm_t1 / gm_t
    except (TypeError, ZeroDivisionError):
        missing_fields.append("GMI")
        gmi = None
    try:
        aqi_t = 1 - (t["current_assets"] + t["ppe"]) / t["total_assets"]
        aqi_t1 = 1 - (t1["current_assets"] + t1["ppe"]) / t1["total_assets"]
        aqi = aqi_t / aqi_t1
    except (TypeError, ZeroDivisionError):
        missing_fields.append("AQI")
        aqi = None
    try:
        sgi = t["sales"] / t1["sales"]
    except (TypeError, ZeroDivisionError):
        missing_fields.append("SGI")
        sgi = None
    try:
        if t["depreciation"] is None or t1["depreciation"] is None:
            raise TypeError
        dep_rate_t = t["depreciation"] / (t["depreciation"] + t["ppe"])
        dep_rate_t1 = t1["depreciation"] / (t1["depreciation"] + t1["ppe"])
        depi = dep_rate_t1 / dep_rate_t
    except (TypeError, ZeroDivisionError):
        missing_fields.append("DEPI(결측->1.0 대체)")
        depi = 1.0
    try:
        sgai = (t["sga"] / t["sales"]) / (t1["sga"] / t1["sales"])
    except (TypeError, ZeroDivisionError):
        missing_fields.append("SGAI")
        sgai = None
    try:
        lvgi = (t["total_liabilities"] / t["total_assets"]) / (t1["total_liabilities"] / t1["total_assets"])
    except (TypeError, ZeroDivisionError):
        missing_fields.append("LVGI")
        lvgi = None
    try:
        dep_for_tata = t["depreciation"] if t["depreciation"] is not None else 0
        wc_t = (t["current_assets"] - t["cash"]) - (t["current_liabilities"])
        wc_t1 = (t1["current_assets"] - t1["cash"]) - (t1["current_liabilities"])
        tata = (wc_t - wc_t1 - dep_for_tata) / t["total_assets"]
    except (TypeError, ZeroDivisionError):
        missing_fields.append("TATA")
        tata = None

    detail = {
        "DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
        "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata,
        "missing_fields": "; ".join(missing_fields) if missing_fields else "",
    }
    core_vals = [dsri, gmi, aqi, sgi, sgai, lvgi, tata]
    if any(v is None for v in core_vals):
        return None, detail

    m_score = (
        -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
        + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    )
    return m_score, detail


def main():
    df = pd.read_csv("control_labels.csv", dtype={"corp_code": str})

    results = []
    for _, row in df.iterrows():
        corp_code = row["corp_code"]
        corp_name = row["corp_name"]
        year = row["year"]

        print(f"조회 중: {corp_name} ({year}년, corp_code={corp_code})")

        t, t1 = fetch_financial_year(corp_code, year, API_KEY, fs_div="CFS")
        if t is None or all(v is None for v in t.values()):
            t, t1 = fetch_financial_year(corp_code, year, API_KEY, fs_div="OFS")

        if t is None:
            print(f"  -> 데이터 조회 실패")
            results.append({"corp_name": corp_name, "corp_code": corp_code,
                             "year": year, "m_score": None, "note": "데이터 없음"})
            time.sleep(0.3)
            continue

        m_score, detail = compute_beneish_m(t, t1)
        if m_score is None:
            print(f"  -> M-Score 계산 실패. 누락 항목: {detail.get('missing_fields', '?')}")
        row_result = {
            "corp_name": corp_name, "corp_code": corp_code, "year": year,
            "m_score": m_score,
        }
        row_result.update(detail)
        results.append(row_result)

        time.sleep(0.3)

    out = pd.DataFrame(results)
    out.to_csv("control_scores.csv", index=False, encoding="utf-8-sig")
    print(f"\n완료. {len(out)}건 처리 -> control_scores.csv 저장")


if __name__ == "__main__":
    main()
