import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import cv2
import easyocr
import joblib
import os
import time
from PIL import Image
from google import genai

# 1. 환경 설정 및 초기화
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr()

# API 설정
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

RF_MODEL_PATH = 'bkt_rf_model.pkl' 
DATA_PATH = 'bkt_training_dataset_english_problem.csv'

# 세션 상태 초기화 (강력한 run_id 생성)
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.final_feedback = []
    st.session_state.run_id = str(time.time()).replace(".", "")

# ===============================
# 2. 핵심 로직 함수
# ===============================

def analyze_answer_logic(question, student_ans):
    """1, 2단계 공통 고품질 채점"""
    prompt = (
        f"문제: {question}\n학생의 답: {student_ans}\n\n"
        "지침:\n"
        "1. 정오답 여부를 첫 줄에 'O' 또는 'X'로 표시.\n"
        "2. 논리적 오류를 상세히 분석.\n"
        "3. 보완이 필요한 수학 개념 설명.\n"
        "4. ✅ 정답: [값] 필수 포함."
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return resp.text
    except:
        return "X\n분석 중 오류가 발생했습니다."

def diagnose_learning_status(results):
    """RF 모델 기반 진단"""
    if not os.path.exists(RF_MODEL_PATH): return "MODEL_MISSING", 0, 0, 0
    try:
        model = joblib.load(RF_MODEL_PATH)
        correctness = [r.get('is_correct', 0) for r in results]
        accuracy = sum(correctness) / len(results) if results else 0
        initial_k = correctness[0] * 0.4 if correctness else 0
        final_k = correctness[-1] * 0.8 if correctness else 0
        variance = np.var(correctness) if len(correctness) > 1 else 0.0
        
        input_df = pd.DataFrame([[initial_k, final_k, max(0, final_k-initial_k), accuracy, variance]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        return model.predict(input_df)[0], accuracy, max(0, final_k-initial_k), initial_k
    except: return "DIAG_ERROR", 0, 0, 0

# ===============================
# 3. UI 및 흐름 제어
# ===============================

if st.session_state.step == 0:
    st.title("🎓 AI BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            if len(pool) >= 15:
                selected_en = random.sample(pool, 15)
                temp_probs = []
                
                # 번역 과정을 완전히 숨기고 결과만 준비
                with st.spinner("학습 세션을 준비하는 중입니다..."):
                    for i, en_text in enumerate(selected_en):
                        try:
                            # 625 문제 등을 포함해 강제 번역 수행
                            ko_text = client.models.generate_content(
                                model=MODEL_NAME, 
                                contents=f"Translate this math problem into Korean only. Use the '-하시오' ending. Never output English. Problem: {en_text}"
                            ).text
                        except:
                            ko_text = en_text
                        
                        temp_probs.append({'id': i+1, 'question': ko_text, 'text_ans': "", 'img_ans': None, 'input_type': '⌨️ 타이핑'})
                
                st.session_state.problems = temp_probs
                st.session_state.run_id = str(time.time()).replace(".", "") # 시작 시 run_id 갱신
                st.session_state.step = 1
                st.rerun()

elif st.session_state.step == 1:
    st.title("📝 1차 학습: 15문제")
    for i, p in enumerate(st.session_state.problems):
        with st.container():
            st.markdown(f"### **Q{p.get('id', i+1)}.**")
            st.info(p.get('question'))
            
            p['input_type'] = st.radio(f"방식(Q{i+1})", ["⌨️ 타이핑", "📸 사진 촬영"], key=f"r_{i}_{st.session_state.run_id}", horizontal=True)
            
            if p['input_type'] == "⌨️ 타이핑":
                # 자동완성 팝업 방지를 위해 라벨에 보이지 않는 고유번호 부여
                unique_label = f"정답 입력 (#{st.session_state.run_id[-4:]}_{i})"
                p['text_ans'] = st.text_input(unique_label, key=f"in_{i}_{st.session_state.run_id}")
            else:
                img = st.camera_input(f"촬영(Q{i+1})", key=f"cam_{i}_{st.session_state.run_id}")
                if img: p['img_ans'] = img
            st.divider()

    if st.button("📤 모든 답안 제출", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 분석 결과")
    if not st.session_state.feedback_results:
        p_bar = st.progress(0)
        for i, p in enumerate(st.session_state.problems):
            ans = p.get('text_ans', "")
            if p.get('input_type') == "📸 사진 촬영" and p.get('img_ans'):
                ans = "이미지 답안 분석됨" # 실제 OCR 처리 로직 연동 가능
            
            res_text = analyze_answer_logic(p.get('question', ''), ans)
            is_correct = 1 if res_text.strip().startswith('O') else 0
            st.session_state.feedback_results.append({'id': p.get('id', i+1), 'is_correct': is_correct, 'content': res_text})
            
            if is_correct == 0:
                new_q = client.models.generate_content(model=MODEL_NAME, contents=f"이 문제('{p.get('question')}')의 원리를 담은 한글 추천 문제를 하나 생성해줘.").text
                st.session_state.new_recommendations.append({'ref_id': p.get('id', i+1), 'new_question': new_q, 'new_ans': ""})
            p_bar.progress((i + 1) / 15)
        p_bar.empty()

    for res in st.session_state.feedback_results:
        if res['is_correct']: st.success(f"✅ Q{res['id']}\n{res['content']}")
        else: st.error(f"❌ Q{res['id']}\n{res['content']}")
    
    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제 풀기", type="primary"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec['new_question'])
        rec['new_ans'] = st.text_input(f"답안 입력 (C:{st.session_state.run_id[-3:]}-{i})", key=f"f_{i}_{st.session_state.run_id}")

    if st.button("🏁 최종 진단", type="primary"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 성취도 리포트")
    if not st.session_state.final_feedback:
        for rec in st.session_state.new_recommendations:
            res = analyze_answer_logic(rec['new_question'], rec['new_ans'])
            st.session_state.final_feedback.append({'ref_id': rec['ref_id'], 'content': res})
        
        status, acc, gain, init = diagnose_learning_status(st.session_state.feedback_results)
        prompt = f"상태:{status}, 정확도:{acc*100:.1f}%. 학생에게 A~E 등급을 부여하고 수학 학습 방향을 한국어로 조언해줘."
        st.session_state.bkt_report = client.models.generate_content(model=MODEL_NAME, contents=prompt).text

    for f in st.session_state.final_feedback:
        st.info(f['content'])
    st.success(st.session_state.bkt_report)
    if st.button("🔄 처음으로"):
        st.session_state.clear()
        st.rerun()
