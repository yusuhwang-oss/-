"""
[최종 단계] 이상치 처리 + 스케일링 + 로지스틱회귀로 "분식회계 확률 예측 모델" 완성
- AICE-Associate 시험에서 배운 흐름(train_test_split -> RobustScaler -> 모델학습 -> 평가)을
  그대로 활용했습니다.

실행 전 준비물: beneish_scores.csv, control_scores.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]

# ---------- 1. 데이터 불러오기 ----------
fraud = pd.read_csv("beneish_scores.csv")
fraud["label"] = 1  # 분식기업

control = pd.read_csv("control_scores.csv")
control["label"] = 0  # 정상기업

df = pd.concat([fraud, control], ignore_index=True)

# ---------- 2. 결측치 처리 ----------
# 8개 지표 중 하나라도 비어있는 행은 분석에서 제외
df = df.dropna(subset=FEATURES)
print(f"결측치 제거 후 표본 수: {len(df)} (분식 {sum(df['label']==1)} / 정상 {sum(df['label']==0)})")

# ---------- 3. 이상치 처리 (Clipping) ----------
# 각 지표별로 1%~99% 범위를 벗어나는 극단값을 경계값으로 눌러줌 (행 삭제 대신 값만 보정)
# -> 표본이 적을 때는 행을 통째로 지우는 것보다 이 방식이 데이터 손실이 적음
for col in FEATURES:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)

print("이상치 클리핑 완료 (상하위 1% 극단값을 경계값으로 조정)")

# ---------- 4. 학습/검증 데이터 분리 ----------
X = df[FEATURES]
y = df["label"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5. 스케일링 (RobustScaler로 이상치 영향 완화) ----------
rs = RobustScaler()
X_train = rs.fit_transform(X_train)
X_valid = rs.transform(X_valid)

# ---------- 6. 로지스틱 회귀 모델 학습 ----------
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# ---------- 7. 성능 평가 ----------
y_pred = model.predict(X_valid)
print("\n=== 검증 성능 ===")
print(f"정확도(Accuracy): {accuracy_score(y_valid, y_pred):.3f}")
print(classification_report(y_valid, y_pred, target_names=["정상", "분식"]))

# ---------- 8. 지표별 영향력 확인 ----------
coef_df = pd.DataFrame({"지표": FEATURES, "계수": model.coef_[0]})
coef_df = coef_df.sort_values("계수", key=abs, ascending=False)
print("=== 지표별 영향력(계수) ===")
print(coef_df.to_string(index=False))


# ---------- 9. 새 회사 데이터 입력 -> 분식 확률(%) 계산 함수 ----------
def predict_fraud_probability(dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata):
    """
    감사대상 회사의 Beneish 8개 지표를 입력하면 분식 확률(%)을 반환.
    (지표 계산 방법은 03_fetch_financials_beneish.py의 compute_beneish_m() 참고)
    """
    x_new = np.array([[dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]])
    x_new_scaled = rs.transform(x_new)  # 학습 때 사용한 스케일러 그대로 적용
    proba = model.predict_proba(x_new_scaled)[0, 1]
    return round(proba * 100, 2)


# ---------- 10. 사용 예시 ----------
if __name__ == "__main__":
    print("\n=== 사용 예시 ===")
    example_prob = predict_fraud_probability(
        dsri=1.5, gmi=1.3, aqi=1.2, sgi=1.4,
        depi=1.0, sgai=0.8, lvgi=1.1, tata=0.05,
    )
    print(f"예시 회사의 분식 확률: {example_prob}%")
