import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import random
from PIL import Image

# 1. 환경 설정 및 모델 로드
st.set_page_config(page_title="AI BKT 카메라 튜터", page_icon="🎓", layout="centered")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
else:
    st.error("❌ API 키를 설정해주세요.")
    st.stop()

@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    return pd.read_csv(data_path) if os.path.exists(data_path) else None

df = load_data()

# 세션 상태 관리
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0       # 0: 시작 화면, 1: 1차 풀이, 2: 채점/추천, 3: 최종 진단
    st.session_state.original_problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []

# ==========================================
# 🏠 [단계 0] 시작 화면 (문제 생성 버튼)
# ==========================================
if st.session_state.current_step == 0:
    st.title("🎓 AI 맞춤형 BKT 학습 시스템")
    st.write("학습 데이터셋에서 15문제를 배정받아 오늘의 학습을 시작하세요.")
    st.divider()
    
    if st.button("🚀 오늘의 문제 생성 및 학습 시작", type="primary", use_container_width=True):
        if df is not None:
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
            pool = df[prob_col].dropna().unique().tolist()
            
            if len(pool) >= 15:
                selected = random.sample(pool, 15)
                st.session_state.original_problems = [
                    {'id': i+1, 'question': q, 'input_type': '타이핑', 'text_ans': "", 'img_ans': None} 
                    for i, q in enumerate(selected)
                ]
                st.session_state.current_step = 1
                st.rerun()
            else:
                st.warning("데이터셋의 문항 수가 부족합니다.")
        else:
            st.error("데이터셋 파일을 찾을 수 없습니다.")

# ==========================================
# 📖 [단계 1] 15문제 풀이 (카메라 촬영/타이핑)
# ==========================================
elif st.session_state.current_step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    st.write("각 문제 아래에서 답변 방식을 선택하세요. 사진은 카메라로 직접 찍어야 합니다.")
    
    for i, p in enumerate(st.session_state.original_problems):
        with st.container():
            st.markdown(f"### **Q{p['id']}.**")
            st.info(p['question'])
            
            p['input_type'] = st.radio(f"답변 방식 (Q{p['id']})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"radio_{i}", horizontal=True)
            
            if p['input_type'] == "⌨️ 타이핑":
                p['text_ans'] = st.text_input("정답 입력", key=f"text_{i}", value=p['text_ans'])
            else:
                img_capture = st.camera_input(f"답안 촬영 (Q{p['id']})", key=f"cam_{i}")
                if img_capture:
                    p['img_ans'] = Image.open(img_capture)
            st.divider()
            
    if st.button("📤 1차 답안 제출 및 채점", type="primary", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()

# ==========================================
# 📊 [단계 2] 채점 결과 및 추천 문제 생성
# ==========================================
elif st.session_state.current_step == 2:
    st.title("🔍 1차 채점 결과 및 맞춤 추천")
    with st.spinner('AI 분석 및 추천 문제 생성 중...'):
        if not st.session_state.feedback_results:
            for p in st.session_state.original_problems:
                student_answer = p['text_ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans']:
                    ocr_res = gemini_model.generate_content(["이 카메라 사진의 답안을 읽어줘.", p['img_ans']])
                    student_answer = ocr_res.text

                grade_prompt = f"문제: {p['question']}\n학생의 답: {student_answer}\n맞으면 'O', 틀리면 'X'와 이유를 적어줘."
                grade_res = gemini_model.generate_content(grade_prompt).text
                is_correct = grade_res.strip().startswith('O')
                
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'feedback': grade_res})

                if not is_correct:
                    gen_prompt = f"이 문제({p['question']})의 개념을 유지하며 숫자나 인수를 바꾼 변형 문제를 1개 만들어줘."
                    new_q = gemini_model.generate_content(gen_prompt).text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'new_question': new_q, 'new_ans': ""})
        
        for res in st.session_state.feedback_results:
            if not res['is_correct']: st.error(f"**[Q{res['id']} 오답]** {res['feedback']}")
            else: st.write(f"✅ Q{res['id']} 정답!")
                
    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제 풀기 (최종 단계)", type="primary"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        st.success("🎉 모든 문제를 맞혔습니다!")
        if st.button("🔄 처음으로"):
            st.session_state.clear()
            st.rerun()

# ==========================================
# 🏆 [단계 3] 추천 문제 풀이 및 최종 진단
# ==========================================
elif st.session_state.current_step == 3:
    st.title("🎯 최종 학습: 오답 맞춤 추천")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.markdown(f"### **변형 문제 (Q{rec['ref_id']} 기반)**")
        st.info(rec['new_question'])
        rec['new_ans'] = st.text_input(f"정답 입력", key=f"rec_ans_{i}")

    if st.button("🏁 최종 답안 제출 및 BKT 진단", type="primary", use_container_width=True):
        with st.spinner('최종 진단 중...'):
            final_data = "\n".join([f"문제: {r['new_question']}, 학생답: {r['new_ans']}" for r in st.session_state.new_recommendations])
            bkt_prompt = f"학생이 추천 문제를 풀었어:\n{final_data}\n\nBKT 분석을 통해 최종 학습 등급(A~E)과 피드백을 줘."
            st.markdown(gemini_model.generate_content(bkt_prompt).text)
            st.balloons()
            
        if st.button("🔄 처음으로"):
            st.session_state.clear()
            st.rerun()

st.caption("LG AI CAMP NEW | Gemini 2.5 Flash | Camera-Only Mode")
