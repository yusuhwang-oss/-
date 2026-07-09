"""
[2-B단계] 한창, 영풍이 동명이인 회사와 중복 매칭된 것을 정리.
- 우리가 찾는 회사는 둘 다 유가증권시장(코스피) 상장사이므로,
  stock_code(종목코드)가 채워진 행만 남기고 나머지는 제거합니다.
"""

import pandas as pd

matched = pd.read_csv("fraud_labels_matched.csv", dtype={"corp_code": str, "stock_code": str})

# 한창, 영풍 중 종목코드가 비어있는(=동명이인 비상장 회사) 행을 제거
mask_bad_duplicate = (
    matched["corp_name"].isin(["한창", "영풍"])
    & (matched["stock_code"].isna() | (matched["stock_code"].str.strip() == ""))
)

cleaned = matched[~mask_bad_duplicate].copy()

print(f"정리 전: {len(matched)}행 -> 정리 후: {len(cleaned)}행")
print("\n[최종 확인] 한창, 영풍 남은 행:")
print(cleaned[cleaned["corp_name"].isin(["한창", "영풍"])][["corp_name", "corp_code", "stock_code"]].to_string(index=False))

# 혹시 남은 중복이 있는지 재확인
dup_check = cleaned["corp_name"].value_counts()
dup_check = dup_check[dup_check > 1]
if len(dup_check) > 0:
    print("\n경고: 아직 중복이 남아있습니다 ->")
    print(dup_check)
else:
    print("\n중복 없음 확인 완료.")

cleaned.to_csv("fraud_labels_matched.csv", index=False, encoding="utf-8-sig")
print(f"\n저장 완료 (덮어쓰기) -> fraud_labels_matched.csv, 최종 {len(cleaned)}개사")
