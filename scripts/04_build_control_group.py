"""
[4단계] 정상기업(비교집단) 표본 구성
- 분식기업 11개와 같은 연도대에서, 분식기업이 아닌 상장기업을 무작위로 추출합니다.
- 분식기업 1개당 정상기업 3개씩 뽑아서(약 1:3 비율), 총 33개 표본을 만듭니다.
- 같은 연도대에서 뽑는 이유: 경기 상황(호황/불황)에 따른 재무비율 왜곡을 줄이기 위함입니다.

실행 전 준비물: beneish_scores.csv (분식기업 결과), dart_corp_code_all.csv
실행하면 -> control_labels.csv 파일이 생성됩니다.
"""

import pandas as pd
import random

random.seed(42)  # 재현 가능하도록 시드 고정

fraud = pd.read_csv("beneish_scores.csv")
fraud_valid = fraud[fraud["m_score"].notna()].copy()  # M-Score 계산 성공한 11개사만 기준

corp_all = pd.read_csv("dart_corp_code_all.csv", dtype={"corp_code": str, "stock_code": str})

# 상장기업만 (정상기업 비교집단은 데이터 확보가 쉬운 상장기업 위주로)
listed = corp_all[corp_all["stock_code"].notna() & (corp_all["stock_code"].str.strip() != "")].copy()

# 분식기업 corp_code는 후보에서 제외
fraud_codes = set(fraud["corp_code"].astype(str))
listed = listed[~listed["corp_code"].isin(fraud_codes)]

# 금융업(은행/보험/지주사 등)은 재무제표 구조가 완전히 달라서 제외 (이름에 은행/보험/금융지주 포함시 제외)
exclude_kw = ["은행", "보험", "금융지주", "캐피탈", "저축은행", "증권"]
listed = listed[~listed["corp_name"].str.contains("|".join(exclude_kw), na=False)]

listed = listed.reset_index(drop=True)

N_PER_FRAUD = 3
control_rows = []
used_codes = set()

for _, frow in fraud_valid.iterrows():
    year = frow["year"]
    # 무작위로 N_PER_FRAUD개 추출 (중복 방지)
    candidates = listed[~listed["corp_code"].isin(used_codes)]
    sample = candidates.sample(n=N_PER_FRAUD, random_state=random.randint(0, 100000))
    for _, srow in sample.iterrows():
        control_rows.append({
            "corp_name": srow["corp_name"],
            "corp_code": srow["corp_code"],
            "year": year,
            "label": 0,  # 정상기업
        })
        used_codes.add(srow["corp_code"])

control_df = pd.DataFrame(control_rows)
print(f"정상기업(비교집단) 표본 수: {len(control_df)}")
print(control_df.head(10).to_string(index=False))

control_df.to_csv("control_labels.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료 -> control_labels.csv")
