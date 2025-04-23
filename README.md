# AI_GenDev

📘 **이 노트북은 데이터 분석 및 머신러닝 학습을 위한 실습 내용을 정리한 것입니다.**


# 📘 KNN 실습 

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

# 📘 Linear, Ridge, Lasso Regression 실습 

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

