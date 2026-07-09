"""
[최종] 기존 정상기업(control_scores.csv) + 신규 정상기업(control_scores_extra.csv)을
       합쳐서 표본을 확대한 뒤, 최종 로지스틱 회귀 모델을 학습합니다.

실행 전 준비물: beneish_scores.csv, control_scores.csv, control_scores_extra.csv
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]

# ---------- 1. 데이터 로드 및 병합 ----------
fraud = pd.read_csv("beneish_scores.csv")
fraud["label"] = 1

control1 = pd.read_csv("control_scores.csv")
control2 = pd.read_csv("control_scores_extra.csv")
control = pd.concat([control1, control2], ignore_index=True)
control["label"] = 0

# 혹시 모를 중복 회사 제거 (같은 corp_code가 두 번 안 뽑히도록)
control = control.drop_duplicates(subset=["corp_code"])

df = pd.concat([fraud, control], ignore_index=True)
df = df.dropna(subset=FEATURES)

print(f"전체 표본: {len(df)} (분식 {sum(df['label']==1)} / 정상 {sum(df['label']==0)})")

# ---------- 2. 이상치 클리핑 ----------
for col in FEATURES:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)

X = df[FEATURES]
y = df["label"]

# ---------- 3. 파이프라인 구성 (스케일링 + 로지스틱회귀) ----------
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])

# ---------- 4. 5-Fold 교차검증으로 성능 확인 ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print("\n=== 5-Fold 교차검증 정확도 ===")
print(f"각 fold별 정확도: {np.round(scores, 3)}")
print(f"평균 정확도: {scores.mean():.3f} (표준편차: {scores.std():.3f})")

auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
print(f"\n평균 AUC: {auc_scores.mean():.3f} (표준편차: {auc_scores.std():.3f})")

# ---------- 5. 최종 모델은 전체 데이터로 학습 (실전 사용) ----------
pipeline.fit(X, y)
print("\n전체 데이터로 최종 모델 학습 완료")

coef_df = pd.DataFrame({
    "지표": FEATURES,
    "계수": pipeline.named_steps["model"].coef_[0],
}).sort_values("계수", key=abs, ascending=False)
print("\n=== 지표별 영향력(계수) ===")
print(coef_df.to_string(index=False))


def predict_fraud_probability(dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata):
    """새 회사의 Beneish 8개 지표를 입력하면 분식 확률(%)을 반환."""
    x_new = pd.DataFrame([{
        "DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
        "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata,
    }])
    proba = pipeline.predict_proba(x_new)[0, 1]
    return round(proba * 100, 2)


if __name__ == "__main__":
    print("\n=== 사용 예시 ===")
    example_prob = predict_fraud_probability(
        dsri=1.5, gmi=1.3, aqi=1.2, sgi=1.4,
        depi=1.0, sgai=0.8, lvgi=1.1, tata=0.05,
    )
    print(f"예시 회사의 분식 확률: {example_prob}%")
