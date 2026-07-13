# 재무비율 기반 분식회계 확률 예측 모델

AICE-Associate 과정에서 배운 데이터 분석/머신러닝 기법을 실제 국내 회계 데이터에 적용해,
**감사대상회사의 재무정보를 입력하면 분식회계 확률(%)을 출력하는 예측 모델**을 개발한 프로젝트입니다.

자세한 배경, 데이터 수집 과정, 결과 해석, 한계점은 [`report.pdf`](./report.pdf)에 정리되어 있습니다.

## 프로젝트 개요

- **목표**: 감사대상회사의 재무비율 8개(Beneish M-Score 지표)를 계산해, 딥러닝 모델로 분식회계 확률을 예측
- **데이터**: 국내 확정 분식회계 기업 11개사 + 정상기업 76개사(중소형 상장사 49개 + 대기업 27개), 총 87개사
- **라벨 출처**: 금융위원회 증권선물위원회(증선위) 보도자료에서 실명이 확인된 확정 조치 사례
- **재무데이터**: DART Open API(전자공시시스템)를 통해 직접 수집
- **모델**: Keras 기반 다층 신경망(Dense-Dropout-Dense-Sigmoid), class_weight로 클래스 불균형 보정

## 사용된 재무비율 (Beneish M-Score 8개 지표)

| 지표 | 의미 |
|---|---|
| DSRI | 매출채권 지수 (매출채권/매출액 비율의 전기 대비 변화) |
| GMI | 매출총이익률 지수 |
| AQI | 자산의 질 지수 |
| SGI | 매출성장률 지수 |
| DEPI | 감가상각률 지수 |
| SGAI | 판관비율 지수 |
| LVGI | 레버리지 지수 |
| TATA | 총발생액 비율 (실제 현금흐름과 회계상 이익의 괴리) |

## 폴더 구조

```
.
├── fraud_prediction_model.ipynb   # 메인 노트북 (이것만 실행하면 됩니다)
├── training_data_final.csv       # 고정된 학습용 데이터 (87개사)
├── fraud_labels.csv              # 분식회계 확정 기업 라벨 원본
├── report.pdf # 상세 설명 보고서
├── requirements.txt
└── scripts/                      # 데이터 수집 과정 스크립트 (참고용, 순서대로 번호가 매겨져 있음)
    ├── 01_get_corp_code.py       # DART 전체 법인 목록 수집
    ├── 02_match_corp_code.py     # 분식기업 라벨과 corp_code 매칭
    ├── 03_fetch_financials_beneish.py  # 분식기업 재무데이터 수집 + 지표 계산
    ├── 04~11 ...                 # 정상기업(비교집단) 수집 및 초기 모델링 시도
    ├── 12_freeze_training_data.py      # 학습 데이터 고정
    ├── 13_add_large_cap_normal.py      # 대기업 표본 보강
    ├── 14_refreeze_training_data.py    # 대기업 포함 후 데이터 재고정
    ├── 15_add_new_fraud_companies.py   # 분식기업 표본 추가 시도
    └── 16_revert_to_87samples.py       # 최종 87개 표본으로 확정
```

`scripts/` 폴더는 실제 개발 과정(시행착오 포함)을 그대로 남겨둔 것입니다.
바로 실행하고 싶다면 `fraud_prediction_model.ipynb`와 `training_data_final.csv`만 있으면 됩니다.

## 실행 방법

### 1) 빠르게 결과만 보고 싶은 경우

```bash
pip install -r requirements.txt
jupyter notebook fraud_prediction_model.ipynb
```

노트북을 위에서부터 순서대로 실행하면 됩니다. 11번 섹션(새 회사 예측)에서는 본인의 DART API 키가 필요합니다.
[DART Open API](https://opendart.fss.or.kr)에서 무료로 발급받을 수 있습니다.

```python
# 노트북 11번 섹션에서
API_KEY = '여기에_본인_DART_API_키를_입력하세요'
target_corp_code = '00126380'  # 예: 삼성전자
target_year = 2022
```

### 2) 데이터 수집 과정부터 재현하고 싶은 경우

`scripts/` 폴더의 파일들을 01번부터 순서대로 실행하면 됩니다. 각 스크립트 상단에 필요한 입력 파일과 설명이 주석으로 달려있습니다.

## 주요 결과 및 한계

- 명확히 건전한 기업(삼성전자, 현대자동차, 빙그레)은 모두 10%대 초반의 낮은 확률로 일관되게 판정
- 확정 분식기업(SOOP)은 그보다 높은 확률로 산출되어 방향성 있는 변별력 확인
- 다만 재무적으로 특이한 시기를 지난 기업(예: 업황 급락기의 반도체 기업)에 대해서는 판단이 불안정한 한계 존재
- 표본 크기(87개, 그중 분식 11개)의 근본적 한계로 인해 실행마다 성능 지표가 다소 변동하는 재현성 이슈 확인

자세한 내용은 보고서의 "5. 모델 결과 및 해석", "6. 실전 테스트 사례" 섹션을 참고해 주세요.

## 사용한 기술 스택

`pandas` `numpy` `scikit-learn` `tensorflow(keras)` `matplotlib` `seaborn` · DART Open API

## 라이선스

이 프로젝트는 학습 목적으로 제작되었으며, 자유롭게 참고하셔도 됩니다 (MIT License).

## 업데이트 (v2)

업종 평균 대비 상대지표를 적용해 모델을 개선했습니다. 자세한 내용은 [v2_industry_adjustment/README_v2.md](./v2_industry_adjustment/README_v2.md)와 [v2_industry_adjustment/report_v2.pdf](./v2_industry_adjustment/report_v2.pdf)를 참고해 주세요.
