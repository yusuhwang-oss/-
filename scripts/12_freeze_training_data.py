"""
[데이터 고정] 지금까지 수집한 분식기업 + 정상기업(기존+신규) 데이터를
             하나의 깔끔한 학습용 CSV로 합쳐서 고정합니다.
             이후 모델링은 이 파일 하나만 가지고 진행합니다 (DART API 재호출 불필요).

실행 전 준비물: beneish_scores.csv, control_scores.csv, control_scores_extra.csv
실행하면 -> training_data_final.csv 파일이 생성됩니다.
"""

import pandas as pd

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]

fraud = pd.read_csv("beneish_scores.csv")
fraud["label"] = 1

control1 = pd.read_csv("control_scores.csv")
control2 = pd.read_csv("control_scores_extra.csv")
control = pd.concat([control1, control2], ignore_index=True)
control = control.drop_duplicates(subset=["corp_code"])
control["label"] = 0

df = pd.concat([fraud, control], ignore_index=True)

# 필요한 컬럼만 깔끔하게 정리
df_final = df[["corp_name", "corp_code", "year", "label"] + FEATURES].copy()

# 결측치(8개 지표 중 하나라도 없는 행) 제거
df_final = df_final.dropna(subset=FEATURES)

print(f"최종 고정 데이터셋: {len(df_final)}개 (분식 {sum(df_final['label']==1)} / 정상 {sum(df_final['label']==0)})")

df_final.to_csv("training_data_final.csv", index=False, encoding="utf-8-sig")
print("저장 완료 -> training_data_final.csv")
print("\n이제부터는 이 파일 하나만 있으면 모델링을 처음부터 다시 할 수 있습니다.")
