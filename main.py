import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import random
import time
from PIL import Image

# 1. 환경 설정
st.set_page_config(page_title="AI BKT 튜터", page_icon="🎓", layout="centered")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
else:
    st.error("❌ API 키가 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    return pd.read_csv(data_path) if os.path.exists(data_path) else None

df = load_data()

# 세션 상태 초기화
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
    st.session_state.original_problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.final_feedback = []
    st.session_state.run_id = str(time.time())

def analyze_answer(question, student_ans):
    prompt = (
        f"문제: {question}\n학생의 답: {student_ans}\n\n"
        "지침:\n"
        "1. 정오답 여부를 첫 줄에 'O' 또는 'X'로 표시할 것.\n"
        "2. 학생의 풀이 과정에서 나타난 논리적 오류나 식의 잘못된 부분을 구체적으로 분석할 것.\n"
        "3. 이 문제를 풀기 위해 보완해야 할 수학적 개념을 설명할 것.\n"
        "4. ✅ 정답: [최종 결과값]을 반드시 포함하여 알려줄 것."
    )
    return gemini_model.generate_content(prompt).text

# ==========================================
# 🏠 [단계 0] 시작 화면
# ==========================================
if st.session_state.current_step == 0:
    st.title("🎓 AI 맞춤형 BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if df is not None:
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else df.columns[0]
            pool = df[prob_col].dropna().unique().tolist()
            
            if len(pool) >= 15:
                # 15개를 먼저 선정만 함
                selected_en = random.sample(pool, 15)
                st.session_state.original_problems = [
                    {'id': i+1, 'question_en': txt, 'question': None, 'text_ans': "", 'img_ans': None, 'input_type': '⌨️ 타이핑'}
                    for i, txt in enumerate(selected_en)
                ]
                st.session_state.current_step = 1
                st.rerun()

# ==========================================
# 📝 [단계 1] 1차 학습 (실시간 번역 출력)
# ==========================================
elif st.session_state.current_step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    
    all_translated = True
    for i, p in enumerate(st.session_state.original_problems):
        with st.container():
            # [수정] 문제가 아직 번역되지 않았다면 이 시점에 번역 (사용자에게 과정 노출 최소화)
            if p['question'] is None:
                try:
                    p['question'] = gemini_model.generate_content(f"Translate this math problem to Korean only, using formal '-하시오' style: {p['question_en']}").text
                except:
                    p['question'] = p['question_en']
            
            st.markdown(f"### **Q{p['id']}.**")
            st.info(p['question'])
            
            p['input_type'] = st.radio(f"답변 방식 (Q{p['id']})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"r_{i}_{st.session_state.run_id}", horizontal=True)
            
            if p['input_type'] == "⌨️ 타이핑":
                p['text_ans'] = st.text_input("정답 입력", key=f"input_{i}_{st.session_state.run_id}", placeholder="답안을 입력하세요.")
            else:
                img = st.camera_input(f"Capture (Q{p['id']})", key=f"cam_{i}_{st.session_state.run_id}")
                if img:
                    p['img_ans'] = Image.open(img)
            st.divider()
            
    if st.button("📤 답안 제출 및 1차 채점", type="primary", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()

# [이후 단계 2, 3, 4는 이전과 동일하되 analyze_answer 함수를 사용하여 일관성 유지]
# (코드 생략 - 위와 동일한 로직 적용)
