"""
[복원] 순서가 정상적으로 나왔던 87개 표본(분식 11 + 정상 76) 구성으로 되돌립니다.
      일양약품, 에스케이엔펄스를 제외합니다.

실행하면 -> beneish_scores.csv에서 두 회사가 제거되고,
          training_data_final.csv도 87개로 재생성됩니다.
"""

import pandas as pd

EXCLUDE_CORP_CODES = ["146454", "307189"]  # 일양약품, 에스케이엔펄스

# ---------- 1. beneish_scores.csv에서 제외 ----------
df = pd.read_csv("beneish_scores.csv", dtype={"corp_code": str})
df["corp_code"] = df["corp_code"].str.zfill(8)
exclude_padded = [c.zfill(8) for c in EXCLUDE_CORP_CODES]

before = len(df)
df_filtered = df[~df["corp_code"].isin(exclude_padded)]
print(f"beneish_scores.csv: {before}행 -> {len(df_filtered)}행 (2개 제외)")
df_filtered.to_csv("beneish_scores.csv", index=False, encoding="utf-8-sig")

# ---------- 2. training_data_final.csv 재생성 (기존 14번 스크립트와 동일 로직) ----------
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]

fraud = pd.read_csv("beneish_scores.csv")
fraud["label"] = 1

control1 = pd.read_csv("control_scores.csv")
control2 = pd.read_csv("control_scores_extra.csv")
control3 = pd.read_csv("control_scores_largecap.csv")
control = pd.concat([control1, control2, control3], ignore_index=True)
control = control.drop_duplicates(subset=["corp_code"])
control["label"] = 0

merged = pd.concat([fraud, control], ignore_index=True)
final = merged[["corp_name", "corp_code", "year", "label"] + FEATURES].copy()
final = final.dropna(subset=FEATURES)

print(f"\n최종 복원된 데이터셋: {len(final)}개 "
      f"(분식 {sum(final['label']==1)} / 정상 {sum(final['label']==0)})")

final.to_csv("training_data_final.csv", index=False, encoding="utf-8-sig")
print("저장 완료 -> training_data_final.csv (87개로 복원됨)")
