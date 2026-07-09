"""
[표본 보강] 시가총액 상위 대기업 30개를 정상기업(label=0) 표본에 추가.
- 기존 정상기업(중소형 상장사 위주)만으로는 초대형 기업 판단력이 떨어지는 문제를 보완하기 위함.
- 분식기업 리스트(18개)와 겹치지 않는 대기업만 선정했습니다.

실행 전 준비물: dart_corp_code_all.csv
실행하면 -> control_labels_largecap.csv, control_scores_largecap.csv 파일이 생성됩니다.
"""

import requests
import pandas as pd
import time

API_KEY = "여기에_본인_DART_API_키를_입력하세요"
BASE_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

# ---------- 1. 대기업 30개 리스트 (분식기업 리스트와 중복 없음) ----------
# 주의: 삼성전자, 현대자동차, SK하이닉스는 검증(held-out test)용으로 남겨두기 위해 의도적으로 제외
LARGE_CAP_NAMES = [
    "기아", "LG전자",
    "삼성바이오로직스", "삼성SDI", "현대모비스", "POSCO홀딩스", "LG화학",
    "삼성물산", "SK이노베이션", "NAVER", "카카오", "KB금융",
    "신한지주", "하나금융지주", "우리금융지주", "셀트리온", "LG생활건강",
    "아모레퍼시픽", "한국전력공사", "SK텔레콤", "KT", "한화솔루션",
    "GS건설", "대한항공", "롯데케미칼", "LG에너지솔루션", "삼성전기",
    "삼성에스디에스", "HD현대중공업", "CJ제일제당",
]
TARGET_YEAR = 2023  # 최근 정상 영업 연도 기준

# ---------- 2. corp_code 매칭 ----------
corp_all = pd.read_csv("dart_corp_code_all.csv", dtype={"corp_code": str, "stock_code": str})
listed = corp_all[corp_all["stock_code"].notna() & (corp_all["stock_code"].str.strip() != "")]

matched_rows = []
for name in LARGE_CAP_NAMES:
    hit = listed[listed["corp_name"] == name]
    if len(hit) == 0:
        print(f"[매칭 실패] {name} -> dart_corp_code_all.csv에서 못 찾음, 수작업 확인 필요")
        continue
    if len(hit) > 1:
        print(f"[중복 주의] {name} -> {len(hit)}건 발견, 첫 번째 것 사용 (수작업 확인 권장)")
    row = hit.iloc[0]
    matched_rows.append({
        "corp_name": name,
        "corp_code": row["corp_code"],
        "year": TARGET_YEAR,
        "label": 0,
    })

labels_df = pd.DataFrame(matched_rows)
print(f"\n대기업 매칭 성공: {len(labels_df)}/{len(LARGE_CAP_NAMES)}개")
labels_df.to_csv("control_labels_largecap.csv", index=False, encoding="utf-8-sig")
print("저장 완료 -> control_labels_largecap.csv")

# ---------- 3. 재무데이터 조회 + Beneish 8개 지표 계산 (결측 대응 포함) ----------
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
    """0으로 나누거나 None이 섞이면 None을 반환하는 안전한 나눗셈"""
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_ratios(t, t1):
    # DSRI
    r1 = safe_div(t["receivables"], t["sales"])
    r2 = safe_div(t1["receivables"], t1["sales"])
    dsri = safe_div(r1, r2)
    if dsri is None:
        dsri = 1.0

    # GMI
    if t["cogs"] is not None and t1["cogs"] is not None and t["sales"] and t1["sales"]:
        gm_t = safe_div(t["sales"] - t["cogs"], t["sales"])
        gm_t1 = safe_div(t1["sales"] - t1["cogs"], t1["sales"])
        gmi = safe_div(gm_t1, gm_t)
        if gmi is None:
            gmi = 1.0
    else:
        gmi = 1.0

    # AQI
    if None not in (t["current_assets"], t["ppe"], t["total_assets"],
                     t1["current_assets"], t1["ppe"], t1["total_assets"]):
        aqi_t = safe_div(t["total_assets"] - t["current_assets"] - t["ppe"], t["total_assets"])
        aqi_t1 = safe_div(t1["total_assets"] - t1["current_assets"] - t1["ppe"], t1["total_assets"])
        aqi = safe_div(aqi_t, aqi_t1)
        if aqi is None:
            aqi = 1.0
    else:
        aqi = 1.0

    # SGI
    sgi = safe_div(t["sales"], t1["sales"])
    if sgi is None:
        sgi = 1.0

    # DEPI
    depi = None
    if t["depreciation"] and t1["depreciation"] and t["ppe"] and t1["ppe"]:
        dep_rate_t = safe_div(t["depreciation"], t["depreciation"] + t["ppe"])
        dep_rate_t1 = safe_div(t1["depreciation"], t1["depreciation"] + t1["ppe"])
        depi = safe_div(dep_rate_t1, dep_rate_t)
    if depi is None:
        depi = 1.0

    # SGAI
    s1 = safe_div(t["sga"], t["sales"])
    s2 = safe_div(t1["sga"], t1["sales"])
    sgai = safe_div(s1, s2)
    if sgai is None:
        sgai = 1.0

    # LVGI
    l1 = safe_div(t["total_liabilities"], t["total_assets"])
    l2 = safe_div(t1["total_liabilities"], t1["total_assets"])
    lvgi = safe_div(l1, l2)
    if lvgi is None:
        lvgi = 1.0

    # TATA
    tata = None
    if None not in (t["current_assets"], t["cash"], t["current_liabilities"], t["total_assets"],
                     t1["current_assets"], t1["cash"], t1["current_liabilities"]) and t["total_assets"]:
        dep_for_tata = t["depreciation"] if t["depreciation"] else 0
        wc_t = (t["current_assets"] - t["cash"]) - t["current_liabilities"]
        wc_t1 = (t1["current_assets"] - t1["cash"]) - t1["current_liabilities"]
        tata = safe_div(wc_t - wc_t1 - dep_for_tata, t["total_assets"])
    if tata is None:
        tata = 0.0

    return {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
            "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata}


results = []
for _, row in labels_df.iterrows():
    corp_code, corp_name, year = row["corp_code"], row["corp_name"], row["year"]
    print(f"조회 중: {corp_name} ({year}년)")
    t, t1 = fetch_financial_year(corp_code, year, fs_div="CFS")
    if t is None or all(v is None for v in t.values()):
        t, t1 = fetch_financial_year(corp_code, year, fs_div="OFS")
    if t is None:
        print("  -> 데이터 조회 실패")
        time.sleep(0.3)
        continue
    ratios = compute_ratios(t, t1)
    result_row = {"corp_name": corp_name, "corp_code": corp_code, "year": year}
    result_row.update(ratios)
    results.append(result_row)
    time.sleep(0.3)

out = pd.DataFrame(results)
out.to_csv("control_scores_largecap.csv", index=False, encoding="utf-8-sig")
print(f"\n완료. {len(out)}건 처리 -> control_scores_largecap.csv 저장")
