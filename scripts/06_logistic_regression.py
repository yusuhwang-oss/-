"""
[6단계] 분식기업(label=1) + 정상기업(label=0) 데이터를 합쳐서
        로지스틱 회귀 모델을 학습하고, 새 회사 데이터를 넣으면
        분식 확률(%)을 출력하는 함수까지 만듭니다.

실행 전 준비물: beneish_scores.csv, control_scores.csv
pip install scikit-learn (없으면 설치: pip install scikit-learn --user)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]

# ---------- 1. 데이터 로드 및 병합 ----------
fraud = pd.read_csv("beneish_scores.csv")
fraud["label"] = 1

control = pd.read_csv("control_scores.csv")
control["label"] = 0

data = pd.concat([fraud, control], ignore_index=True)

# 8개 지표가 전부 있는 행만 사용 (계산 실패/결측 제외)
data_clean = data.dropna(subset=FEATURES).copy()

print(f"전체 표본: {len(data)} (분식 {len(fraud)} + 정상 {len(control)})")
print(f"모델 학습에 쓸 수 있는(결측없는) 표본: {len(data_clean)}")
print(f"  - 분식(label=1): {(data_clean['label']==1).sum()}개")
print(f"  - 정상(label=0): {(data_clean['label']==0).sum()}개")

if data_clean['label'].nunique() < 2 or (data_clean['label']==1).sum() < 3:
    print("\n경고: 표본이 너무 적어서 안정적인 학습이 어려울 수 있습니다.")
    print("      (분식기업 표본이 특히 적으면 결과 해석에 주의가 필요합니다)")

# ---------- 2. 학습/검증 분리 ----------
X = data_clean[FEATURES]
y = data_clean["label"]

# 표본이 적으므로 test_size를 작게, stratify로 label 비율 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------- 3. 로지스틱 회귀 학습 ----------
model = LogisticRegression(max_iter=1000, class_weight="balanced")  # 표본 불균형 보정
model.fit(X_train, y_train)

# ---------- 4. 성능 평가 ----------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== 테스트셋 성능 ===")
print(classification_report(y_test, y_pred, target_names=["정상", "분식"]))
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.3f}")
except ValueError:
    print("AUC 계산 불가 (테스트셋에 한 클래스만 존재)")

# ---------- 5. 지표별 계수(중요도) 확인 ----------
coef_df = pd.DataFrame({
    "지표": FEATURES,
    "계수": model.coef_[0],
}).sort_values("계수", key=abs, ascending=False)
print("\n=== 지표별 영향력(계수) ===")
print(coef_df.to_string(index=False))
print("\n(계수의 절대값이 클수록 분식 확률 예측에 영향력이 큰 지표입니다)")


# ---------- 6. 새 회사 데이터 입력 -> 분식 확률(%) 출력 함수 ----------
def predict_fraud_probability(dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata):
    """
    Beneish 8개 지표를 입력하면 분식 확률(%)을 반환.
    각 지표는 02/03/05번 스크립트의 compute_beneish_m()과 동일한 방식으로 계산해야 함.
    """
    x_new = pd.DataFrame([{
        "DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
        "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata,
    }])
    proba = model.predict_proba(x_new)[0, 1]
    return round(proba * 100, 2)


# ---------- 사용 예시 ----------
if __name__ == "__main__":
    print("\n=== 사용 예시: 새 감사대상회사 데이터 입력 ===")
    example_prob = predict_fraud_probability(
        dsri=1.5, gmi=1.3, aqi=1.2, sgi=1.4,
        depi=1.0, sgai=0.8, lvgi=1.1, tata=0.05,
    )
    print(f"예시 회사의 분식 확률: {example_prob}%")
    print("\n실제 사용시: DART에서 감사대상 회사의 당기/전기 재무제표를 가져와서")
    print("위 8개 지표를 직접 계산한 뒤 predict_fraud_probability()에 넣으면 됩니다.")
