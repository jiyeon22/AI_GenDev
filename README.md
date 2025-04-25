# AI_GenDev

📘 **이 노트북은 데이터 분석 및 머신러닝 학습을 위한 실습 내용을 정리한 것입니다.**


# 📘1. KNN 실습 

K-최근접 이웃 알고리즘(K-Nearest Neighbors)을 다양한 데이터셋에 적용해보며  
회귀 및 분류 문제에 대한 모델 성능, 데이터 스케일링의 중요성, 예측 결과 해석 등을 실습한 자료입니다.

## 📁 파일
- 📄 [`knn_project.ipynb`](./knn_project.ipynb)  
  → KNN 알고리즘을 이용한 회귀/분류 실습, 스케일링 효과 비교, 붓꽃 분류 등 포함

## 🧪 실습 주제
- KNN 회귀 (부동산 가격 예측)
- KNN 분류 (붓꽃 아이리스 분류)
- 데이터 스케일링 전후 성능 비교
- 다양한 k 값에 따른 정확도 변화 시각화

# 📘 2. Linear, Ridge, Lasso Regression 실습 

선형 회귀(Linear Regression)와 정규화 기법인 Ridge, Lasso 모델을 이용해
연속형 타겟 변수 예측을 수행하고, **과적합 방지**, **모델 해석**, **성능 비교** 등을 실습한 자료입니다.

## 📁 파일
- 📄 [`Linear_Ridge_Lasso.ipynb`](./Linear_Ridge_Lasso.ipynb)  
  → 선형 회귀 계열 모델 적용, 성능 지표 비교, 가중치 변화 시각화, 규제 효과 분석 등 포함

## 🧪 실습 주제
- 선형 회귀 (Linear Regression)
- 릿지 회귀 (Ridge Regression, L2 규제)
- 라쏘 회귀 (Lasso Regression, L1 규제)
- 회귀 성능 지표 비교
- alpha 값 변화에 따른 가중치 변화 시각화
- 과적합 방지 효과 확인

## 💡 주요 포인트
- Ridge: 모든 피처 사용, **가중치만 줄여서 과적합 억제**
- Lasso: **불필요한 피처는 제거 (가중치 0으로)** → 변수 선택 효과
- 단순 선형 회귀보다 일반화 성능 향상

# 📘 3. 고객 이탈 예측 실습 - Decision Tree & SVM 모델 비교

고객 이탈 데이터셋을 기반으로, 의사결정트리(Decision Tree)와 서포트 벡터 머신(SVM) 모델을 활용하여  
이탈 여부를 예측하고, 두 모델의 성능을 비교 분석한 실습 자료입니다.

## 📁 파일
- 📄 [`DecisionTree_SVM_Telco_Churn_Assignment_Only.ipynb`](./DecisionTree_SVM_Telco_Churn_Assignment_Only.ipynb)  
  → 고객 이탈 데이터셋을 활용한 전처리, 의사결정트리 & SVM 모델 비교, 성능 분석 및 시각화 포함

## 🧪 실습 주제
- 고객 이탈(`Churn`)을 타겟으로 한 **이진 분류(Binary Classification) 문제**
- **의사결정트리(DecisionTreeClassifier)** 와 **SVM(Support Vector Classifier)**을 적용
- 모델 성능을 **정확도, 정밀도(Precision), 재현율(Recall), F1-score**로 비교
- 결정트리 시각화를 통한 **주요 분기 기준 해석**
- 샘플 1건에 대한 예측 결과 비교 (두 모델이 동일하게 판단하는지 확인)

## 주요 전처리 내용

- `TotalCharges` 컬럼의 문자열 → 수치형 변환 및 결측치 처리
- `Churn` 컬럼 이진화 (`Yes` → 1, `No` → 0)
- `customerID` 제거 (식별자)
- 범주형 변수 → **원-핫 인코딩** (`get_dummies`)
- `train_test_split()`을 사용하여 **학습/테스트 데이터 분할**
- 결측치 제거 (`dropna()`) 

## 학습 소감
이번 실습을 통해 모델 성능을 단순한 정확도 외에도 정밀도, 재현율 등 **다양한 평가 지표를 활용해 해석하는 경험**을 할 수 있었습니다.  
특히 의사결정트리 시각화를 통해 **모델이 어떤 기준으로 분기하고 판단하는지 직접 확인**할 수 있었고,  
**상황에 따라 어떤 모델을 선택해야 할지 고민해보는 시간**이 되었습니다.



