import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import google.generativeai as genai
import nest_asyncio
import random

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 지능형 튜토리얼", page_icon="🎓", layout="wide")

# 2. 보안 설정: Secrets로부터 API 키 로드
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    st.error("❌ API 키를 찾을 수 없습니다. Streamlit Secrets 설정을 확인해주세요.")
    st.stop()

# 3. 리소스 로드 함수
@st.cache_resource
def load_resources():
    model_path = 'bkt_rf__model.pkl'
    data_path = 'bkt_training_dataset_english_problem.csv'
    
    rf_model = joblib.load(model_path) if os.path.exists(model_path) else None
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
        
    return rf_model, df

rf_model, df = load_resources()

# 4. 사이드바: 학생 데이터 관리
st.sidebar.title("👤 학생 관리")
if df is not None:
    student_list = sorted(df['user_id'].unique())
    selected_student = st.sidebar.selectbox("진단 대상 학생 ID", student_list)
    student_history = df[df['user_id'] == selected_student]
    student_data = student_history.iloc[-1]
else:
    st.sidebar.error("데이터셋 파일을 찾을 수 없습니다.")
    st.stop()

# 5. 메인 화면 구성
st.title("🎓 AI 지능형 학습 진단 및 맞춤형 문제 추천")
st.write(f"현재 **학생 ID: {selected_student}**의 학습 데이터를 분석 중입니다.")

# 대시보드 요약
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("최근 결과", "✅ 정답" if student_data['correct'] == 1 else "❌ 오답")
with col2:
    st.metric("응답 속도", f"{student_data['ms_first_response']/1000:.2f}초")
with col3:
    st.metric("총 소요 시간", f"{student_data['overlap_time']/1000:.2f}초")
with col4:
    st.metric("현재 진도", f"{student_data['order_id']}회차")

st.divider()

# 6. 탭 구성
tab1, tab2, tab3 = st.tabs(["🧐 AI 정밀 진단", "📚 맞춤형 15문제 추천", "📊 학습 이력"])

# --- 탭 1: AI 진단 및 피드백 ---
with tab1:
    st.subheader("🤖 BKT 모델 및 Gemini AI 통합 분석")
    if st.button("실시간 진단 실행"):
        if rf_model is not None:
            # RF 모델 입력 (5 features)
            features = np.array([[
                student_data['correct'],
                student_data['ms_first_response'],
                student_data['overlap_time'],
                student_data['first_action'],
                student_data['order_id']
            ]])
            prediction = rf_model.predict(features)[0]
            status = "숙달(Mastery)" if prediction == 1 else "미숙달(Non-Mastery)"
            
            with st.spinner('Gemini가 분석 메시지를 생성 중입니다...'):
                prompt = f"학생 학습 데이터(결과:{status}, 시간:{student_data['ms_first_response']}ms)를 바탕으로 3줄 내외의 격려 피드백을 한글로 작성해줘."
                response = gemini_model.generate_content(prompt)
            
            st.info(f"📊 **BKT 진단 결과:** 현재 학생은 **{status}** 상태입니다.")
            st.success(f"💌 **AI 피드백:**\n\n{response.text}")
            if prediction == 1: st.balloons()
        else:
            st.error("모델 파일을 찾을 수 없습니다.")

# --- 탭 2: 중복 없는 15문제 제공 (신규 추가 로직) ---
with tab2:
    st.subheader("📝 학생 맞춤형 연습 문제 (15문항)")
    st.write("데이터셋에서 중복되지 않은 15개의 문제를 무작위로 추출합니다.")
    
    if st.button("새로운 문제 세트 생성"):
        # 전체 데이터셋에서 고유한 문제들 추출 (이미지 기반 컬럼 활용)
        # 'generated_problem_english' 또는 'problem_id' 기준
        if 'generated_problem_english' in df.columns:
            all_problems = df['generated_problem_english'].dropna().unique().tolist()
        else:
            all_problems = df['problem_id'].dropna().unique().tolist()

        if len(all_problems) >= 15:
            # 중복 없이 15개 무작위 샘플링
            selected_problems = random.sample(all_problems, 15)
            
            for i, prob in enumerate(selected_problems, 1):
                with st.expander(f"Q{i}. 문제 확인하기"):
                    st.write(prob)
                    st.text_input(f"Q{i} 정답 입력", key=f"answer_{i}")
            
            st.success("✅ 15문제가 성공적으로 생성되었습니다. 문제를 풀고 제출하세요!")
        else:
            st.warning(f"현재 가용한 고유 문제가 {len(all_problems)}개뿐입니다. 모든 문제를 표시합니다.")
            for i, prob in enumerate(all_problems, 1):
                st.write(f"Q{i}. {prob}")

# --- 탭 3: 상세 이력 ---
with tab2:
    st.subheader(f"📈 학생 ID {selected_student}의 전체 로그")
    st.dataframe(student_history)
