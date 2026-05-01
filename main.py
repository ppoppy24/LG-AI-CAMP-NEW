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

# [핵심] 동일한 채점 로직을 위한 공통 함수
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
# 🏠 [단계 0] 시작 화면 (번역 과정 표시 제거)
# ==========================================
if st.session_state.current_step == 0:
    st.title("🎓 AI 맞춤형 BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if df is not None:
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else df.columns[0]
            pool = df[prob_col].dropna().unique().tolist()
            
            if len(pool) >= 15:
                selected_en = random.sample(pool, 15)
                temp_problems = []
                # 번역 과정을 사용자에게 노출하지 않고 내부적으로 즉시 처리
                for i, en_text in enumerate(selected_en):
                    try:
                        # "번역만" 수행하여 한글 텍스트만 추출
                        ko_text = gemini_model.generate_content(f"Translate this to Korean only: {en_text}").text
                    except:
                        ko_text = en_text
                    temp_problems.append({
                        'id': i+1, 'question': ko_text, 
                        'text_ans': "", 'img_ans': None, 'input_type': '⌨️ 타이핑'
                    })
                
                st.session_state.original_problems = temp_problems
                st.session_state.current_step = 1
                st.rerun()

# ==========================================
# 📝 [단계 1] 1차 학습 (15문제)
# ==========================================
elif st.session_state.current_step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    
    for i, p in enumerate(st.session_state.original_problems):
        with st.container():
            st.markdown(f"### **Q{p['id']}.**")
            st.info(p['question'])
            
            p['input_type'] = st.radio(f"답변 방식 (Q{p['id']})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"r_{i}_{st.session_state.run_id}", horizontal=True)
            
            if p['input_type'] == "⌨️ 타이핑":
                p['text_ans'] = st.text_input("정답 입력", key=f"input_{i}_{st.session_state.run_id}")
            else:
                img = st.camera_input(f"Capture (Q{p['id']})", key=f"cam_{i}_{st.session_state.run_id}")
                if img:
                    p['img_ans'] = Image.open(img)
            st.divider()
            
    if st.button("📤 답안 제출 및 1차 채점", type="primary", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()

# ==========================================
# 📊 [단계 2] 1차 상세 분석 결과
# ==========================================
elif st.session_state.current_step == 2:
    st.title("🔍 1차 채점 및 논리 분석")
    
    with st.spinner('AI가 답안을 분석 중입니다...'):
        if not st.session_state.feedback_results:
            for p in st.session_state.original_problems:
                ans = p['text_ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans']:
                    ans = gemini_model.generate_content(["이미지 속 답안 텍스트를 읽어줘.", p['img_ans']]).text

                res_text = analyze_answer(p['question'], ans)
                is_correct = res_text.strip().startswith('O')
                
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'content': res_text})

                if not is_correct:
                    # 변형 문제 생성 (번역 과정 없이 바로 한글로 생성)
                    new_q = gemini_model.generate_content(f"이 문제('{p['question']}')의 개념을 담은 한글 변형 문제를 하나 만들어줘.").text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'new_question': new_q, 'new_ans': ""})
        
        for res in st.session_state.feedback_results:
            if res['is_correct']:
                st.success(f"✅ **Q{res['id']} 정답**\n\n{res['content']}")
            else:
                st.error(f"❌ **Q{res['id']} 오답 분석**\n\n{res['content']}")
            st.divider()
                
    if st.session_state.new_recommendations:
        if st.button("🚀 분석 기반 추천 문제 풀기", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

# ==========================================
# 🎯 [단계 3] 2차 추천 문제 풀이
# ==========================================
elif st.session_state.current_step == 3:
    st.title("🎯 2차 학습: 맞춤 추천 문제")
    st.caption("1차 학습에서 오답이 발생한 개념을 보완하기 위한 문제입니다.")
    
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.markdown(f"### **추천 문제 (Q{rec['ref_id']} 보완)**")
        st.info(rec['new_question'])
        rec['new_ans'] = st.text_input(f"답 입력 (Q{rec['ref_id']}-1)", key=f"final_{i}_{st.session_state.run_id}")
        st.divider()

    if st.button("🏁 최종 제출 및 진단", type="primary", use_container_width=True):
        st.session_state.current_step = 4
        st.rerun()

# ==========================================
# 🏆 [단계 4] 최종 결과 및 BKT 진단 (1단계와 동일한 피드백 수준)
# ==========================================
elif st.session_state.current_step == 4:
    st.title("🏆 최종 학습 진단")
    
    with st.spinner('최종 성취도를 분석 중입니다...'):
        if not st.session_state.final_feedback:
            for rec in st.session_state.new_recommendations:
                # 2단계 피드백도 1단계와 동일한 함수 사용하여 공평하게 제공
                res_text = analyze_answer(rec['new_question'], rec['new_ans'])
                st.session_state.final_feedback.append({'ref_id': rec['ref_id'], 'content': res_text})
            
            # 전체 데이터를 기반으로 한 BKT 진단
            summary_data = "\n".join([f"Q: {r['new_question']}, A: {r['new_ans']}" for r in st.session_state.new_recommendations])
            st.session_state.bkt_report = gemini_model.generate_content(f"다음 풀이 이력을 보고 BKT 등급(A~E)과 학습 처방을 내려줘: {summary_data}").text

        for res in st.session_state.final_feedback:
            st.markdown(f"#### 📝 추천 문제 분석 (Q{res['ref_id']} 기반)")
            st.info(res['content'])
            st.divider()

        st.success("### 📊 종합 BKT 성취도 리포트")
        st.markdown(st.session_state.bkt_report)
        st.balloons()
            
        if st.button("🔄 다시 시작하기"):
            st.session_state.clear()
            st.rerun()
