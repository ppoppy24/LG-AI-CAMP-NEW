import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import random
from PIL import Image

# ==========================================
# ⚙️ 초기 설정 및 환경 구성
# ==========================================
st.set_page_config(page_title="AI 단계별 맞춤형 튜터", page_icon="🎓", layout="centered")

# Gemini 2.5 Flash 모델 로드 (에러 시 2.0으로 변경)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
else:
    st.error("❌ API 키를 설정해주세요.")
    st.stop()

# 데이터셋 로드
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    return pd.read_csv(data_path) if os.path.exists(data_path) else None

df = load_data()

# ==========================================
# 🧠 세션 상태 관리 (학습 흐름 제어)
# ==========================================
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1       # 현재 단계 (1~3)
    st.session_state.original_problems = [] # 1차 문제 리스트
    st.session_state.feedback_results = []  # 1차 채점 결과
    st.session_state.new_recommendations = [] # 추천(변형) 문제 리스트

# ==========================================
# 📖 [단계 1~3] 15문제 배정 및 입력 방식 선택 후 풀이
# ==========================================
if st.session_state.current_step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    st.write("각 문제마다 원하는 방식(사진 또는 타이핑)으로 답안을 제출해 주세요.")
    
    # [1] 데이터셋에서 중복 없는 15문제 배정
    if not st.session_state.original_problems:
        if df is not None:
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
            pool = df[prob_col].dropna().unique().tolist()
            selected = random.sample(pool, 15) if len(pool) >= 15 else pool
            
            st.session_state.original_problems = [
                {'id': i+1, 'question': q, 'input_type': '타이핑', 'text_ans': "", 'img_ans': None} 
                for i, q in enumerate(selected)
            ]
            st.rerun()
        else:
            st.error("데이터셋을 찾을 수 없습니다.")

    # [2, 3] 각 문제당 입력 칸 생성 및 답안 작성
    if st.session_state.original_problems:
        for i, p in enumerate(st.session_state.original_problems):
            with st.container():
                st.markdown(f"### **Q{p['id']}.**")
                st.info(p['question'])
                
                # 입력 방식 선택
                p['input_type'] = st.radio(f"입력 방식 선택 (Q{p['id']})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"radio_{i}", horizontal=True)
                
                # 선택한 방식에 따른 입력창 제공
                if p['input_type'] == "⌨️ 타이핑":
                    p['text_ans'] = st.text_input("정답 입력", key=f"text_{i}", value=p['text_ans'])
                else:
                    img_file = st.file_uploader("답안 사진 업로드", type=['jpg', 'jpeg', 'png'], key=f"img_{i}")
                    if img_file:
                        p['img_ans'] = Image.open(img_file)
                        st.image(p['img_ans'], width=250, caption="업로드된 답안")
                st.divider()
                
        if st.button("📤 1차 답안 제출 및 채점", type="primary", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()

# ==========================================
# 📊 [단계 4~5] 1차 채점/피드백 및 오답 변형 문제 생성
# ==========================================
elif st.session_state.current_step == 2:
    st.title("🔍 1차 채점 결과 및 맞춤 추천")
    
    with st.spinner('AI가 답안을 분석하고 맞춤형 문제를 생성 중입니다...'):
        if not st.session_state.feedback_results:
            wrong_count = 0
            for p in st.session_state.original_problems:
                # 사진 답안인 경우 OCR로 텍스트화
                student_answer = p['text_ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans']:
                    ocr_res = gemini_model.generate_content(["이 이미지의 손글씨나 텍스트를 읽어줘.", p['img_ans']])
                    student_answer = ocr_res.text

                # [4] 채점 및 피드백 (틀린 문제만 상세 피드백)
                grade_prompt = f"문제: {p['question']}\n학생의 답: {student_answer}\n이 답이 맞았는지 틀렸는지 'O' 또는 'X'로 시작하고, 틀렸다면 그 이유를 1~2줄로 설명해줘."
                grade_res = gemini_model.generate_content(grade_prompt).text
                
                is_correct = grade_res.strip().startswith('O')
                st.session_state.feedback_results.append({
                    'id': p['id'], 'question': p['question'], 'student_ans': student_answer, 
                    'is_correct': is_correct, 'feedback': grade_res
                })

                # [5] 틀린 문제에서 인수/조건을 변경하여 새로운 문제 생성
                if not is_correct:
                    wrong_count += 1
                    gen_prompt = f"이 문제({p['question']})를 학생이 틀렸어. 수학이라면 숫자나 인수를 변경하고, 영어라면 단어나 문법 요소를 살짝 바꿔서 같은 개념을 묻는 **새로운 추천 문제**를 1개 만들어줘. 문제 내용만 출력해."
                    new_q = gemini_model.generate_content(gen_prompt).text
                    st.session_state.new_recommendations.append({
                        'ref_id': p['id'], 'new_question': new_q, 'new_ans': ""
                    })
        
        # 화면 출력
        st.success(f"채점 완료! 총 15문제 중 {len(st.session_state.new_recommendations)}문제를 틀렸습니다.")
        for res in st.session_state.feedback_results:
            if not res['is_correct']:
                st.error(f"**[Q{res['id']} 오답]** {res['feedback']}")
            else:
                st.write(f"✅ Q{res['id']} 정답!")
                
    st.divider()
    if st.session_state.new_recommendations:
        st.warning("틀린 개념을 확실히 잡기 위해, 아래 버튼을 눌러 변형된 추천 문제를 풀어보세요!")
        if st.button("🚀 추천 문제 풀기 (최종 단계)", type="primary"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        st.balloons()
        st.success("🎉 만점입니다! 추가로 풀 추천 문제가 없습니다.")
        if st.button("🔄 처음부터 다시 하기"):
            st.session_state.clear()
            st.rerun()

# ==========================================
# 🏆 [단계 6] 추천 문제 풀이 및 최종 BKT 등급 분류
# ==========================================
elif st.session_state.current_step == 3:
    st.title("🎯 최종 학습: 오답 맞춤형 추천 문제")
    st.write("앞서 틀렸던 문제들의 변형 문제입니다. 다시 한번 도전해 보세요!")
    
    # 추천 문제 풀이
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.markdown(f"### **변형 문제 (Q{rec['ref_id']} 기반)**")
        st.info(rec['new_question'])
        rec['new_ans'] = st.text_input(f"정답 입력", key=f"rec_ans_{i}")
        st.divider()

    # 최종 제출 및 BKT 분석
    if st.button("🏁 최종 답안 제출 및 BKT 진단 받기", type="primary", use_container_width=True):
        with st.spinner('최종 채점 및 BKT 기반 학습 등급을 산출하고 있습니다...'):
            # 추천 문제에 대한 채점 정보 정리
            final_data = ""
            for rec in st.session_state.new_recommendations:
                final_data += f"문제: {rec['new_question']}\n학생 최종답안: {rec['new_ans']}\n---\n"
                
            # [6] 최종 BKT 분석 프롬프트
            bkt_prompt = f"""
            학생이 1차에서 틀렸던 개념을 기반으로 변형된 다음 문제들을 다시 풀었어:
            {final_data}
            
            이 답안들을 채점하고, 학생의 지식 상태 변화(Bayesian Knowledge Tracing)를 분석해줘.
            최종적으로 이 학생의 학습 등급(A~E 또는 레벨)을 분류하고, 앞으로 어떤 부분을 더 공부해야 하는지 3줄 이내의 피드백을 작성해줘.
            """
            final_diagnosis = gemini_model.generate_content(bkt_prompt).text
            
            st.balloons()
            st.subheader("📋 BKT 최종 학습 진단 보고서")
            st.markdown(final_diagnosis)
            
        if st.button("🔄 전체 학습 종료 및 처음으로"):
            st.session_state.clear()
            st.rerun()

# 하단 정보
st.caption("LG AI CAMP NEW | 6단계 BKT 기반 자동화 튜터링 시스템")
