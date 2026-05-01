import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import joblib
import os
import time
from PIL import Image
from google import genai

# 1. 환경 설정 및 초기화
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

# API 설정 (Streamlit Cloud Secrets 필수)
API_KEY = st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

RF_MODEL_PATH = 'bkt_rf_model.pkl' 
DATA_PATH = 'bkt_training_dataset_english_problem.csv'

# 세션 상태 초기화 (더 강력한 무작위 ID)
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.run_id = str(int(time.time()))

# ===============================
# 2. 핵심 로직 함수
# ===============================

def safe_translate(en_text):
    """영문 문제를 한글로 강제 번역하는 로직"""
    prompt = f"너는 수학 선생님이야. 다음 영문 문제를 반드시 한국어 '-하시오' 체로만 번역해줘. 영어를 그대로 출력하면 안 돼. 문제: {en_text}"
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        ko_text = resp.text.strip()
        # AI가 영문을 그대로 뱉었는지 확인 (숫자/기호 제외)
        if ko_text.lower() == en_text.lower():
             return f"{re.search(r'\d+', en_text).group() if re.search(r'\d+', en_text) else ''}을 소인수분해하시오."
        return ko_text
    except:
        return en_text

def diagnose_learning_status(results):
    """RF 모델 기반 학습 상태 진단"""
    if not os.path.exists(RF_MODEL_PATH): return "MODEL_FILE_NOT_FOUND", 0, 0
    try:
        model = joblib.load(RF_MODEL_PATH)
        correctness = [r.get('is_correct', 0) for r in results]
        accuracy = sum(correctness) / len(results) if results else 0
        init_k = correctness[0] * 0.4 if correctness else 0
        final_k = correctness[-1] * 0.8 if correctness else 0
        variance = np.var(correctness) if len(correctness) > 1 else 0.0
        
        input_df = pd.DataFrame([[init_k, final_k, max(0, final_k-init_k), accuracy, variance]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        return model.predict(input_df)[0], accuracy, max(0, final_k-init_k)
    except: return "ERROR", 0, 0

# ===============================
# 3. 단계별 UI 로직
# ===============================

if st.session_state.step == 0:
    st.title("🎓 AI 맞춤형 BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            if len(pool) >= 15:
                selected_en = random.sample(pool, 15)
                temp_probs = []
                
                with st.spinner("문제를 한국어로 변환하고 있습니다..."):
                    for i, en_text in enumerate(selected_en):
                        ko_text = safe_translate(en_text)
                        temp_probs.append({'id': i+1, 'question': ko_text, 'ans': ""})
                
                st.session_state.problems = temp_probs
                st.session_state.run_id = str(int(time.time())) # 매 시작마다 ID 갱신
                st.session_state.step = 1
                st.rerun()
        else:
            st.error("데이터셋 파일이 저장소에 없습니다.")

elif st.session_state.step == 1:
    st.title("📝 1차 학습: 15문제")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{p.get('id', i+1)}.**")
        st.info(p.get('question'))
        
        # 자동완성 방지용 무작위 라벨 (브라우저를 속이는 핵심 장치)
        input_label = f"정답을 입력하세요 (세션ID:{st.session_state.run_id[-4:]}-{i})"
        p['ans'] = st.text_input(input_label, key=f"ans_{i}_{st.session_state.run_id}")
        st.divider()

    if st.button("📤 답안 제출 및 분석", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 채점 결과 및 AI 피드백")
    if not st.session_state.feedback_results:
        with st.spinner("AI 선생님이 채점 중입니다..."):
            for p in st.session_state.problems:
                prompt = f"문제: {p.get('question')}\n학생 답: {p.get('ans')}\n판정(O/X), 분석, 개념보강, 정답공개 순으로 상세히 설명해줘."
                resp = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
                is_correct = 1 if resp.strip().startswith('O') else 0
                st.session_state.feedback_results.append({'id': p.get('id'), 'is_correct': is_correct, 'content': resp})
                
                if not is_correct:
                    rec_q = client.models.generate_content(model=MODEL_NAME, contents=f"'{p.get('question')}'의 유사 한글 문제를 하나 만들어줘.").text
                    st.session_state.new_recommendations.append({'ref_id': p.get('id'), 'q': rec_q, 'ans': ""})

    for res in st.session_state.feedback_results:
        if res['is_correct']: st.success(f"✅ Q{res['id']}\n{res['content']}")
        else: st.error(f"❌ Q{res['id']}\n{res['content']}")

    if st.session_state.new_recommendations:
        if st.button("🚀 오답 보완 문제 풀기", type="primary"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec.get('q'))
        rec['ans'] = st.text_input(f"추천 정답 입력 (ID:{st.session_state.run_id[-3:]}-{i})", key=f"rec_{i}_{st.session_state.run_id}")

    if st.button("🏁 최종 학습 진단", type="primary"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 최종 성취도 진단")
    status, acc, gain = diagnose_learning_status(st.session_state.feedback_results)
    
    report_prompt = f"상태:{status}, 정확도:{acc*100:.1f}%. 이 데이터를 근거로 학생에게 등급(A~E)과 학습 처방을 한국어로 내려줘."
    report = client.models.generate_content(model=MODEL_NAME, contents=report_prompt).text
    
    st.success(f"### 분석 결과: {status}")
    st.markdown(report)
    
    if st.button("🔄 처음으로 돌아가기"):
        st.session_state.clear()
        st.rerun()
