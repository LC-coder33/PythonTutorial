import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.float_format = '{:.2f}'.format

# 1. 데이터 로드 및 전처리
df = pd.read_csv('dataset/Crop_recommendation.csv')

# 특성과 타겟 분리
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 레이블 인코딩
le = LabelEncoder()
y = le.fit_transform(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 랜덤포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 모델과 스케일러 저장
joblib.dump(model, 'crop_model.pkl')
joblib.dump(scaler, 'crop_scaler.pkl')
joblib.dump(le, 'crop_encoder.pkl')

# 3. Streamlit 앱
st.title('🌱 스마트 작물 추천 시스템')
st.write('토양 조건과 환경 조건을 입력하면 최적의 작물을 추천해드립니다.')

# 입력 섹션
st.subheader('토양 영양분 조건')
col1, col2, col3 = st.columns(3)
with col1:
    n = st.number_input('질소(N)', min_value=0, max_value=120, value=50)
with col2:
    p = st.number_input('인(P)', min_value=0, max_value=140, value=53)
with col3:
    k = st.number_input('칼륨(K)', min_value=0, max_value=200, value=48)

st.subheader('환경 조건')
col4, col5 = st.columns(2)
with col4:
    temp = st.slider('온도(°C)', min_value=0.0, max_value=45.0, value=26.0)
    humidity = st.slider('습도(%)', min_value=0.0, max_value=100.0, value=71.0)
with col5:
    ph = st.slider('pH', min_value=0.0, max_value=10.0, value=7.0)
    rainfall = st.slider('강수량(mm)', min_value=0.0, max_value=300.0, value=103.0)

# 예측하기 버튼
if st.button('작물 추천받기'):
    # 모델 및 관련 객체 로드
    model = joblib.load('crop_model.pkl')
    scaler = joblib.load('crop_scaler.pkl')
    le = joblib.load('crop_encoder.pkl')
    
    # 입력값 전처리
    input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    
    # 예측 및 확률 계산
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # 상위 3개 작물 추천
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    
    # 결과 출력
    st.subheader('추천 결과')
    
    # 메인 추천 작물
    st.write(f'🏆 최적 추천 작물: **{le.inverse_transform([prediction])[0]}**')
    
    # 상위 3개 작물 및 확률
    st.write('상위 3개 추천 작물:')
    for idx in top_3_idx:
        crop_name = le.inverse_transform([idx])[0]
        probability = probabilities[idx]
        st.write(f'- {crop_name}: {probability:.1%}')
    
    # 신뢰도 게이지
    st.subheader('추천 신뢰도')
    st.progress(probabilities[prediction])
    
    
#N: 80
#P: 45
#K: 40
#temperature: 23
#humidity: 82
#ph: 6.5
#rainfall: 240
# rice

#N: 100
#P: 75
#K: 50
#temperature: 27
#humidity: 80
#ph: 6.5
#rainfall: 100