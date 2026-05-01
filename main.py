import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import nest_asyncio

# 초기 설정
nest_asyncio.apply()
st.set_page_config(page_title="BKT 학습 진단 시스템", page_icon="📈")

# 1. 모델 및 데이터 로드 함수
@st.cache_resource
def load_data_and_model():
    model_path = 'bkt_rf__model.pkl'
    data_path = 'bkt_training_dataset_english_problem.csv'
    
    model = None
    df = None
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
    return model, df

model, df = load_data_and_model()

# 2. 사이드바 - 사용자 입력
st.sidebar.header("📊 학생 데이터 입력")
if df is not None:
    # 데이터셋에 기반한 학생 선택 (예시: 첫 10명만)
    student_list = df['user_id'].unique()[:10] if 'user_id' in df.columns else ["학생1", "학생2"]
    selected_student = st.sidebar.selectbox("대상 학생 선택", student_list)
else:
    selected_student = st.sidebar.text_input("학생 이름 입력", "Guest")

st.sidebar.divider()
st.sidebar.write("💡 **실시간 진단 설정**")
input_study_time = st.sidebar.slider("학습 시간 (분)", 0, 180, 45)
input_score = st.sidebar.slider("최근 성적", 0, 100, 75)

# 3. 메인 화면
st.title("🎓 BKT 기반 학습 진단 시스템")
st.write(f"현재 **{selected_student}** 학생의 학습 패턴을 분석 중입니다.")

tab1, tab2, tab3 = st.tabs(["🧐 학습 진단", "📚 데이터 통계", "📷 문제 인식"])

with tab1:
    st.header("Random Forest 진단 결과")
    
    if st.button("진단 실행"):
        if model is not None:
            # 모델의 Feature 순서에 맞춰 입력값 구성 (예시: 시간, 점수)
            # 주의: 실제 모델 학습 시 사용한 Column 순서와 동일해야 합니다.
            features = np.array([[input_study_time, input_score]])
            
            try:
                prediction = model.predict(features)
                st.success(f"예측된 학습 등급/상태: **{prediction[0]}**")
                
                if prediction[0] == 1 or prediction[0] == 'High':
                    st.balloons()
                    st.write("우수한 성취도를 보이고 있습니다! 심화 문제를 추천합니다.")
                else:
                    st.info("보충 학습이 필요한 단계입니다. 기초 개념을 복습해 보세요.")
            except Exception as e:
                st.error(f"진단 중 오류 발생: {e}")
        else:
            st.error("`bkt_rf__model.pkl` 파일을 찾을 수 없습니다. 깃허브에 업로드했는지 확인해 주세요.")

with tab2:
    st.header("학습 데이터셋 분석")
    if df is not None:
        st.write("업로드된 `bkt_training_dataset_english_problem.csv` 데이터 요약:")
        st.dataframe(df.head(10)) # 데이터 상위 10개 표시
        
        # 간단한 시각화
        if 'score' in df.columns:
            st.subheader("성적 분포")
            st.bar_chart(df['score'].value_counts())
    else:
        st.warning("데이터셋 파일(.csv)을 찾을 수 없습니다.")

with tab3:
    st.header("문제지 스캔")
    img_file = st.camera_input("문제지를 촬영하세요")
    if img_file:
        st.image(img_file, caption="인식된 이미지")
        st.write("텍스트 추출 중... (EasyOCR 작동 준비 완료)")

# 하단 정보
st.divider()
st.caption("LG AI CAMP NEW - Bayesian Knowledge Tracing System v1.1")
