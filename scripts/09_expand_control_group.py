"""
[표본 확장 1단계] 기존 정상기업(control_labels.csv)과 겹치지 않게,
                   정상기업 후보를 추가로 더 뽑습니다.

실행 전 준비물: beneish_scores.csv, control_labels.csv(기존), dart_corp_code_all.csv
실행하면 -> control_labels_extra.csv 파일이 생성됩니다.
"""

import pandas as pd
import random

random.seed(7)  # 기존과 다른 시드로 새로운 표본 추출

fraud = pd.read_csv("beneish_scores.csv")
fraud_valid = fraud[fraud["m_score"].notna()].copy()

existing_control = pd.read_csv("control_labels.csv", dtype={"corp_code": str})

corp_all = pd.read_csv("dart_corp_code_all.csv", dtype={"corp_code": str, "stock_code": str})
listed = corp_all[corp_all["stock_code"].notna() & (corp_all["stock_code"].str.strip() != "")].copy()

# 분식기업 + 기존에 뽑았던 정상기업은 후보에서 제외 (중복 방지)
fraud_codes = set(fraud["corp_code"].astype(str))
existing_codes = set(existing_control["corp_code"].astype(str))
exclude_codes = fraud_codes | existing_codes

listed = listed[~listed["corp_code"].isin(exclude_codes)]

# 금융업 제외
exclude_kw = ["은행", "보험", "금융지주", "캐피탈", "저축은행", "증권"]
listed = listed[~listed["corp_name"].str.contains("|".join(exclude_kw), na=False)]
listed = listed.reset_index(drop=True)

N_PER_FRAUD = 6  # 기존 3개 -> 6개로 늘려서 표본 확대
control_rows = []
used_codes = set()

for _, frow in fraud_valid.iterrows():
    year = frow["year"]
    candidates = listed[~listed["corp_code"].isin(used_codes)]
    sample = candidates.sample(n=N_PER_FRAUD, random_state=random.randint(0, 100000))
    for _, srow in sample.iterrows():
        control_rows.append({
            "corp_name": srow["corp_name"],
            "corp_code": srow["corp_code"],
            "year": year,
            "label": 0,
        })
        used_codes.add(srow["corp_code"])

control_extra_df = pd.DataFrame(control_rows)
print(f"신규 정상기업(비교집단) 표본 수: {len(control_extra_df)}")

control_extra_df.to_csv("control_labels_extra.csv", index=False, encoding="utf-8-sig")
print("저장 완료 -> control_labels_extra.csv")
