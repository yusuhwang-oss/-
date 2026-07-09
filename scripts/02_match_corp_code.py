"""
[2단계] fraud_labels.csv의 회사명을 dart_corp_code_all.csv와 매칭해서
        corp_code를 붙이는 코드.

실행 전 준비물: fraud_labels.csv, dart_corp_code_all.csv (1단계 결과물)
실행하면 -> fraud_labels_matched.csv 파일이 생성됩니다.
"""

import pandas as pd
import re

labels = pd.read_csv("fraud_labels.csv")
corp_all = pd.read_csv("dart_corp_code_all.csv", dtype={"corp_code": str, "stock_code": str})


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name)
    name = name.replace("㈜", "").replace("(주)", "")
    name = re.sub(r"주식회사", "", name)
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"[^\w가-힣]", "", name)
    return name.strip()


labels["norm_name"] = labels["corp_name"].apply(normalize_name)
corp_all["norm_name"] = corp_all["corp_name"].apply(normalize_name)

merged = labels.merge(
    corp_all[["corp_code", "corp_name", "stock_code", "norm_name"]],
    on="norm_name",
    how="left",
    suffixes=("", "_dart"),
)

matched = merged[merged["corp_code"].notna()]
unmatched = merged[merged["corp_code"].isna()]

print(f"전체 라벨 기업 수: {len(labels)}")
print(f"매칭 성공: {len(matched)}")
print(f"매칭 실패: {len(unmatched)}")

if len(unmatched) > 0:
    print("\n[매칭 실패 목록]")
    print(unmatched[["corp_name"]].to_string(index=False))

merged.to_csv("fraud_labels_matched.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료 -> fraud_labels_matched.csv")
