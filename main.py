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

if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
    st.session_state.original_problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.run_id = str(time.time())

# ==========================================
# 🏠 [단계 0] 시작 화면
# ==========================================
if st.session_state.current_step == 0:
    st.title("🎓 AI 맞춤형 BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 15개 생성", type="primary", use_container_width=True):
        if df is not None:
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else df.columns[0]
            pool = df[prob_col].dropna().unique().tolist()
            
            if len(pool) >= 15:
                selected_en = random.sample(pool, 15)
                temp_problems = []
                bar = st.progress(0)
                for i, en_text in enumerate(selected_en):
                    try:
                        ko_text = gemini_model.generate_content(f"Translate to Korean: {en_text}").text
                    except:
                        ko_text = en_text
                    temp_problems.append({
                        'id': i+1, 'question_en': en_text, 'question_ko': ko_text, 
                        'text_ans': "", 'img_ans': None, 'input_type': '⌨️ 타이핑'
                    })
                    bar.progress((i + 1) / 15)
                
                st.session_state.original_problems = temp_problems
                st.session_state.current_step = 1
                st.rerun()

# ==========================================
# 📖 [단계 1] 15문제 풀이
# ==========================================
elif st.session_state.current_step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    
    for i, p in enumerate(st.session_state.original_problems):
        with st.container():
            st.markdown(f"### **Q{p['id']}.**")
            st.info(p['question_ko'])
            with st.expander("영어 원문 보기"):
                st.write(p['question_en'])
            
            p['input_type'] = st.radio(f"답변 방식 (Q{p['id']})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"r_{i}_{st.session_state.run_id}", horizontal=True)
            
            if p['input_type'] == "⌨️ 타이핑":
                p['text_ans'] = st.text_input("정답 입력", key=f"input_{i}_{st.session_state.run_id}", placeholder="답안을 입력하세요.")
            else:
                img = st.camera_input(f"Capture (Q{p['id']})", key=f"cam_{i}_{st.session_state.run_id}")
                if img:
                    p['img_ans'] = Image.open(img)
            st.divider()
            
    if st.button("📤 답안 제출 및 채점", type="primary", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()

# ==========================================
# 📊 [단계 2] 채점 및 논리 피드백 (정답 비공개)
# ==========================================
elif st.session_state.current_step == 2:
    st.title("🔍 채점 결과 및 상세 분석")
    
    with st.spinner('AI가 답안의 논리를 분석 중입니다...'):
        if not st.session_state.feedback_results:
            for p in st.session_state.original_problems:
                student_answer = p['text_ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans']:
                    student_answer = gemini_model.generate_content(["이미지 속 답안을 읽어줘.", p['img_ans']]).text

                # [프롬프트 수정] 정답 공개 금지, 오답 이유와 식의 오류 설명 강조
                grade_prompt = (
                    f"문제: {p['question_ko']}\n학생의 답: {student_answer}\n\n"
                    "지침:\n"
                    "1. 맞으면 'O', 틀리면 'X'를 첫 줄에 출력할 것.\n"
                    "2. 틀린 경우, 학생의 답이나 식에서 어떤 부분이 잘못되었는지 논리적인 이유를 자세히 설명할 것.\n"
                    "3. 학생이 놓치고 있는 개념이 무엇인지 짚어줄 것.\n"
                    "4. 학생에게 답을 알려줄 것"
                )
                res_text = gemini_model.generate_content(grade_prompt).text
                is_correct = res_text.strip().startswith('O')
                
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'content': res_text})

                if not is_correct:
                    gen_prompt = f"이 문제({p['question_ko']})의 개념을 유지하되 숫자나 인수를 바꾼 변형 문제를 한글로 만들어줘."
                    new_q = gemini_model.generate_content(gen_prompt).text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'new_question': new_q, 'new_ans': ""})
        
        # 정오답 결과 표시
        for res in st.session_state.feedback_results:
            if not res['is_correct']:
                # 틀린 문제만 상세 피드백 출력
                st.error(f"❌ **Q{res['id']} 오답 분석**")
                st.write(res['content'])
            else:
                st.success(f"✅ **Q{res['id']}** 정답입니다!")
            st.divider()
                
    if st.session_state.new_recommendations:
        if st.button("🚀 피드백을 바탕으로 추천 문제 풀기", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

# ==========================================
# 🏆 [단계 3] 최종 추천 문제 및 BKT 진단
# ==========================================
elif st.session_state.current_step == 3:
    st.title("🎯 최종 학습: 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.markdown(f"### **변형 문제 (Q{rec['ref_id']} 기반)**")
        st.info(rec['new_question'])
        rec['new_ans'] = st.text_input(f"답 입력", key=f"final_{i}_{st.session_state.run_id}")

    if st.button("🏁 최종 제출 및 진단", type="primary", use_container_width=True):
        with st.spinner('학습 상태를 진단 중입니다...'):
            final_data = "\n".join([f"Q: {r['new_question']}, Ans: {r['new_ans']}" for r in st.session_state.new_recommendations])
            bkt_res = gemini_model.generate_content(f"학생의 추천 문제 풀이를 보고 BKT 기반 등급(A~E)과 학습 방향을 제시해줘: {final_data}").text
            st.markdown(bkt_res)
            st.balloons()
            
        if st.button("🔄 처음으로"):
            st.session_state.clear()
            st.rerun()
