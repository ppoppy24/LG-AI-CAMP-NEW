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
    # 데이터셋 경로 확인
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

# 공통 채점 및 피드백 프롬프트 함수
def get_grade_prompt(question, student_ans):
    return (
        f"문제: {question}\n학생의 답: {student_ans}\n\n"
        "지침:\n"
        "1. 맞으면 'O', 틀리면 'X'를 첫 줄에 출력.\n"
        "2. 학생의 풀이 과정에서 나타난 논리적 오류를 상세히 분석하여 설명.\n"
        "3. 보완이 필요한 수학적 개념을 명시.\n"
        "4. ✅ 정답: [값] 형식으로 정확한 답을 반드시 포함할 것."
    )

# ==========================================
# 🏠 [단계 0] 시작 화면 (번역 과정 제거, 즉시 로드)
# ==========================================
if st.session_state.current_step == 0:
    st.title("🎓 AI 맞춤형 BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 15개 즉시 생성", type="primary", use_container_width=True):
        if df is not None:
            # 데이터셋에서 무작위 15개 추출 (번역 API 호출 없이 즉시 로드)
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else df.columns[0]
            pool = df[prob_col].dropna().unique().tolist()
            
            if len(pool) >= 15:
                selected_probs = random.sample(pool, 15)
                temp_problems = []
                for i, prob_text in enumerate(selected_probs):
                    temp_problems.append({
                        'id': i+1, 
                        'question': prob_text, 
                        'text_ans': "", 
                        'img_ans': None, 
                        'input_type': '⌨️ 타이핑'
                    })
                
                st.session_state.original_problems = temp_problems
                st.session_state.current_step = 1
                st.rerun()

# ==========================================
# 📖 [단계 1] 1차 풀이 (사진/타이핑 선택)
# ==========================================
elif st.session_state.current_step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    
    for i, p in enumerate(st.session_state.original_problems):
        with st.container():
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

# ==========================================
# 📊 [단계 2] 1차 채점 결과 및 상세 분석
# ==========================================
elif st.session_state.current_step == 2:
    st.title("🔍 1차 채점 분석 결과")
    
    with st.spinner('AI가 답안을 정밀 분석 중입니다...'):
        if not st.session_state.feedback_results:
            for p in st.session_state.original_problems:
                student_answer = p['text_ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans']:
                    student_answer = gemini_model.generate_content(["이미지 속 답안을 텍스트로 읽어줘.", p['img_ans']]).text

                # 상세 피드백 및 정답 제공
                res_text = gemini_model.generate_content(get_grade_prompt(p['question'], student_answer)).text
                is_correct = res_text.strip().startswith('O')
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'content': res_text})

                if not is_correct:
                    # 변형 문제 생성
                    new_q = gemini_model.generate_content(f"이 문제({p['question']})와 유사한 변형 문제를 하나 만들어줘.").text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'new_question': new_q, 'new_ans': ""})
        
        for res in st.session_state.feedback_results:
            if not res['is_correct']:
                st.error(f"❌ **Q{res['id']} 오답 분석**\n{res['content']}")
            else:
                st.success(f"✅ **Q{res['id']} 정답입니다!\n{res['content']}")
            st.divider()
                
    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제 풀기 (2단계)", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

# ==========================================
# 🎯 [단계 3] 2차 추천 문제 풀이 및 동일한 피드백
# ==========================================
elif st.session_state.current_step == 3:
    st.title("🎯 2차 학습: 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.markdown(f"### **추천 문제 (Q{rec['ref_id']} 기반)**")
        st.info(rec['new_question'])
        rec['new_ans'] = st.text_input(f"답 입력", key=f"final_{i}_{st.session_state.run_id}")

    if st.button("🏁 최종 제출 및 진단", type="primary", use_container_width=True):
        st.session_state.current_step = 4
        st.rerun()

# ==========================================
# 🏆 [단계 4] 최종 결과 및 BKT 진단
# ==========================================
elif st.session_state.current_step == 4:
    st.title("🏆 최종 학습 진단 결과")
    
    with st.spinner('최종 성취도를 계산 중입니다...'):
        if not st.session_state.final_feedback:
            all_rec_data = ""
            for rec in st.session_state.new_recommendations:
                # 2단계 문제도 1단계와 동일하게 상세 분석 수행
                res_text = gemini_model.generate_content(get_grade_prompt(rec['new_question'], rec['new_ans'])).text
                st.session_state.final_feedback.append({'ref_id': rec['ref_id'], 'content': res_text})
                all_rec_data += f"Q: {rec['new_question']}, Ans: {rec['new_ans']}\n"

            # BKT 기반 종합 진단
            bkt_summary = gemini_model.generate_content(f"학습자의 전체 풀이 데이터를 바탕으로 BKT 성취 등급(A~E)과 최종 조언을 줘: {all_rec_data}").text
            st.session_state.bkt_summary = bkt_summary

        for res in st.session_state.final_feedback:
            st.info(f"📝 **추천 문제(Q{res['ref_id']} 기반) 분석**\n{res['content']}")
        
        st.success("### 📊 최종 BKT 진단 리포트")
        st.markdown(st.session_state.bkt_summary)
        st.balloons()
            
        if st.button("🔄 처음으로"):
            st.session_state.clear()
            st.rerun()
