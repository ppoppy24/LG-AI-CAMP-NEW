import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import google.generativeai as genai

# 1. 초기 설정 및 API 연결
st.set_page_config(page_title="AI 튜토리얼 시스템", page_icon="🤖", layout="wide")

# Gemini API 설정 (본인의 API 키를 입력하세요)
# Streamlit Cloud 배포 시에는 st.secrets["GEMINI_API_KEY"] 사용 권장
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-pro')

# 2. 리소스 로드 (캐싱 적용)
@st.cache_resource
def load_resources():
    model_path = 'bkt_rf__model.pkl'
    data_path = 'bkt_training_dataset_english_problem.csv'
    
    rf_model = joblib.load(model_path) if os.path.exists(model_path) else None
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    
    return rf_model, df

rf_model, df = load_resources()

# 3. 사이드바: 학생 선택
st.sidebar.title("👤 학생 관리")
if df is not None:
    student_list = sorted(df['user_id'].unique())
    selected_student = st.sidebar.selectbox("진단할 학생 ID", student_list)
    student_data = df[df['user_id'] == selected_student].iloc[-1]
else:
    st.error("데이터셋을 찾을 수 없습니다.")
    st.stop()

# 4. 메인 화면
st.title("🎓 AI 지능형 학습 피드백 시스템")
st.write(f"학생 **{selected_student}**의 최신 학습 데이터를 바탕으로 분석을 시작합니다.")

# 지표 요약 시각화
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("첫 응답 속도", f"{student_data['ms_first_response']/1000:.1f}초")
with col2:
    st.metric("총 소요 시간", f"{student_data['overlap_time']/1000:.1f}초")
with col3:
    st.metric("정답 여부", "✅ 정답" if student_data['correct'] == 1 else "❌ 오답")

st.divider()

# 5. 탭 구성
tab1, tab2 = st.tabs(["🎯 BKT AI 진단 및 피드백", "📊 학습 이력 확인"])

with tab1:
    if st.button("AI 정밀 진단 및 피드백 생성"):
        if rf_model is not None:
            # A. Random Forest (BKT) 진단 실행
            # 입력: [correct, ms_first_response, overlap_time, first_action, order_id]
            features = np.array([[
                student_data['correct'],
                student_data['ms_first_response'],
                student_data['overlap_time'],
                student_data['first_action'],
                student_data['order_id']
            ]])
            
            prediction = rf_model.predict(features)[0]
            status = "숙달(Mastery)" if prediction == 1 else "미숙달(Non-Mastery)"
            
            # B. Gemini AI에게 전달할 프롬프트 구성
            prompt = f"""
            너는 전문 교육 상담가야. 아래 학생의 학습 데이터를 보고 따뜻하고 구체적인 피드백을 한글로 작성해줘.
            
            [학생 학습 데이터]
            - 정답 여부: {'정답' if student_data['correct'] == 1 else '오답'}
            - 첫 응답까지 걸린 시간: {student_data['ms_first_response']/1000}초
            - 총 고민 시간: {student_data['overlap_time']/1000}초
            - AI 모델 판정 결과: {status}
            - 학습 진도 순서(order_id): {student_data['order_id']}
            
            [피드백 포함 내용]
            1. 현재 학습 상태에 대한 격려
            2. 응답 시간과 정답 여부를 조합한 분석 (예: 너무 빨리 풀어서 실수했는지, 고민을 많이 했는지)
            3. 앞으로의 학습 방향 제안
            """
            
            # C. 화면 출력
            st.subheader("🤖 AI 분석 결과")
            with st.spinner('Gemini AI가 피드백을 생성 중입니다...'):
                response = gemini_model.generate_content(prompt)
                
                # 결과 박스 디자인
                st.info(f"**BKT 모델 판정:** 이 학생은 현재 **{status}** 상태입니다.")
                st.markdown("---")
                st.markdown(f"### 💌 맞춤형 학습 피드백\n{response.text}")
                
        else:
            st.error("BKT 모델 파일이 없습니다.")

with tab2:
    st.subheader(f"학생 {selected_student}의 전체 기록")
    st.dataframe(df[df['user_id'] == selected_student])
