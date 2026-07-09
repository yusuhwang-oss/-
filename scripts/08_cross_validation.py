"""
[검증 보강] 표본이 작을 때(29개) 한 번의 train/test 분할만 보면
           운에 따라 성능이 과장/과소평가될 수 있습니다.
           교차검증(Cross-Validation)으로 더 안정적인 성능을 확인합니다.

실행 전 준비물: beneish_scores.csv, control_scores.csv
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]

# ---------- 1. 데이터 로드 및 전처리 (이전과 동일) ----------
fraud = pd.read_csv("beneish_scores.csv")
fraud["label"] = 1
control = pd.read_csv("control_scores.csv")
control["label"] = 0

df = pd.concat([fraud, control], ignore_index=True)
df = df.dropna(subset=FEATURES)

for col in FEATURES:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)

X = df[FEATURES]
y = df["label"]

print(f"전체 표본: {len(df)} (분식 {sum(y==1)} / 정상 {sum(y==0)})")

# ---------- 2. 스케일링 + 모델을 하나의 파이프라인으로 묶기 ----------
# (교차검증에서는 매 fold마다 "학습용 데이터에만" 스케일러를 새로 맞춰야
#  데이터 누수(leakage)가 없습니다. Pipeline이 이걸 자동으로 처리해줍니다)
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])

# ---------- 3. 5-Fold 교차검증 ----------
# 표본이 29개뿐이라 5-fold면 fold당 약 5~6개씩 나뉩니다
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print("\n=== 5-Fold 교차검증 정확도 ===")
print(f"각 fold별 정확도: {np.round(scores, 3)}")
print(f"평균 정확도: {scores.mean():.3f} (표준편차: {scores.std():.3f})")
print("\n해석: 평균 정확도가 실제 모델 성능에 더 가까운 추정치입니다.")
print("      fold별 편차가 크다면(표준편차가 크다면), 표본이 적어 결과가 불안정하다는 뜻입니다.")

# ---------- 4. 최종 모델은 전체 데이터로 다시 학습 (실전 배포용) ----------
pipeline.fit(X, y)
print("\n전체 데이터로 최종 모델 학습 완료 (predict_fraud_probability에서 사용)")


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
