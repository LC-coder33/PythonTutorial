import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# 페이지 설정
st.set_page_config(
    page_title="🧠 뇌종양 분류 시스템",
    page_icon="🏥",
    layout="wide"
)

# 제목
st.title("🧠 뇌종양 MRI 분류 시스템")
st.write("MRI 이미지를 업로드하면 종양의 유형을 분석해드립니다.")

# 모델 및 관련 설정 로드
@st.cache_resource
def load_classification_model():
    model = load_model('./model/brain_tumor_classification_model.h5')
    return model

def preprocess_image(image):
    # 이미지 크기 조정
    img = image.resize((150, 150))
    # 배열로 변환
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 배치 차원 추가
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_image(image, model):
    # 클래스 이름 정의
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # 이미지 전처리
    processed_image = preprocess_image(image)
    
    # 예측
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    
    return class_names, score

# 사이드바에 설명 추가
with st.sidebar:
    st.header("📋 분류 가능한 종양 유형")
    st.write("""
    1. Glioma (신경교종)
    2. Meningioma (수막종)
    3. No Tumor (종양 없음)
    4. Pituitary (뇌하수체 종양)
    """)
    
    st.header("ℹ️ 사용 방법")
    st.write("""
    1. MRI 이미지 파일을 업로드합니다.
    2. 자동으로 분석이 시작됩니다.
    3. 분석 결과와 확률을 확인합니다.
    """)

# 메인 영역
try:
    model = load_classification_model()
    
    # 파일 업로더
    uploaded_file = st.file_uploader("MRI 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # 이미지 로드 및 표시
        image = Image.open(uploaded_file)
        
        # 두 컬럼으로 나누기
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("업로드된 MRI 이미지")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("분석 결과")
            
            # 예측 수행
            class_names, score = predict_image(image, model)
            
            # 결과 시각화
            top_pred_idx = np.argmax(score)
            top_pred_name = class_names[top_pred_idx]
            top_pred_score = 100 * np.max(score)
            
            # 메인 예측 결과
            st.markdown(f"### 🏥 진단 결과")
            st.markdown(f"**{top_pred_name.upper()}**")
            
            # 신뢰도 게이지
            st.markdown("### 📊 신뢰도")
            st.progress(float(np.max(score)))
            st.write(f"{top_pred_score:.2f}%")
            
            # 전체 예측 확률
            st.markdown("### 🔍 상세 분석")
            for i in range(len(class_names)):
                st.write(f"{class_names[i]}: {100 * score[i]:.2f}%")

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.error("모델 파일이 올바른 경로에 있는지 확인해주세요.")

# 페이지 하단
st.markdown("---")
st.markdown("© 2025 Human Brain Tumor Classification System. For educational purposes only.")