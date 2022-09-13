---
layout: post
title: "Bike Sharing Demand"
excerpt: "[kaggel] Bike Sharing Demand 분석"

categories:
	- EDA
tags:
	- EDA
last_modified_at:
---
# [Bike Sharing Demand]  
## 목적 : 데이터분석과 시각화, 머신러닝 알고리즘으로 시간당 자전거 대여량을 예측  
  
  1. 목적 
  Kaggle의 Bike Sharing Demand Competition으로 진행한 분석 프로젝트입니다.
  워싱턴 D.C 소재의 자전거 대여 스타트업 [Capital Bikeshare](https://www.capitalbikeshare.com/)의 데이터를 활용하여, 특정 시간대에 얼마나 많은 사람들이 자전거를 대여하는지 예측하는 것이 목표입니다. 
  
  2. 주요 사용언어 및 라이브러리
  - 언어: Python
  - 전처리/분석: Pandas
  - 시각화: matplotlib, Seaborn
  - 머신러닝: scikit-learn
  
  3. 데이터 개요
  데이터는 하기 경로에서 다운로드 가능합니다.
  [Kaggle Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)
  **datetime** - 시간. 연-월-일 시:분:초 로 표현
  **season** - 계절. 봄(1), 여름(2), 가을(3), 겨울(4) 순으로 표현
  **holiday** - 공휴일 여부. 1:공휴일
  **workingday** - 근무일 여부. 1:근무일
  **weather** - 날씨. 1 ~ 4 사이의 값.
  * 1: 아주 깨끗한 날씨입니다. 또는 아주 약간의 구름이 끼어있습니다.
  * 2: 약간의 안개와 구름이 끼어있는 날씨입니다.
  * 3: 약간의 눈, 비가 오거나 천둥이 칩니다.
  * 4: 아주 많은 비가 오거나 우박이 내립니다.
  **temp** - 온도. 섭씨(Celsius)표기.
  **atemp** - 체감 온도.섭씨(Celsius)표기.
  **humidity** - 습도.
  **windspeed** - 풍속.
  **casual** - 비회원(non-registered)의 자전거 대여량.
  **registered** - 회원(registered)의 자전거 대여량.
  **count** - 총 자전거 대여랑. 비회원(casual) + 회원(registered)과 동일.
    
   4. 진행 순서
   - 데이터 확인
   - 데이터 전처리
   - 데이터 분석
   - 가설 수립
   - 가설 검증
   - 모델 트레이닝
   - 하이퍼파라미터 설정
   - 예측 실행
   - 테스트 데이터 정확도 확인
 
