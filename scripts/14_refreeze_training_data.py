"""
[데이터 재고정] 기존 분식기업 + 기존 정상기업(중소형) + 신규 대기업 30개를
              모두 합쳐서 training_data_final.csv를 다시 만듭니다.
              (기존 파일을 덮어씁니다)

실행 전 준비물: beneish_scores.csv, control_scores.csv, control_scores_extra.csv,
              control_scores_largecap.csv
"""

import pandas as pd

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]

fraud = pd.read_csv("beneish_scores.csv")
fraud["label"] = 1

control1 = pd.read_csv("control_scores.csv")
control2 = pd.read_csv("control_scores_extra.csv")
control3 = pd.read_csv("control_scores_largecap.csv")

control = pd.concat([control1, control2, control3], ignore_index=True)
control = control.drop_duplicates(subset=["corp_code"])
control["label"] = 0

df = pd.concat([fraud, control], ignore_index=True)
df_final = df[["corp_name", "corp_code", "year", "label"] + FEATURES].copy()
df_final = df_final.dropna(subset=FEATURES)

print(f"최종 고정 데이터셋: {len(df_final)}개 "
      f"(분식 {sum(df_final['label']==1)} / 정상 {sum(df_final['label']==0)})")
print(f"  - 정상기업 중 대기업 포함: {control3['corp_name'].nunique()}개 시도, "
      f"실제 반영: {len(df_final[df_final['corp_code'].isin(control3['corp_code'])])}개")

df_final.to_csv("training_data_final.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료 (덮어쓰기) -> training_data_final.csv")
print("이제 노트북(최종_분식회계예측모델.ipynb)을 1번 셀부터 다시 실행하면 됩니다.")
