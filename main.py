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

# 1. 환경 설정
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

# API 설정 (Streamlit Cloud Secrets 필수)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# 파일 경로 (GitHub Root 기준)
RF_MODEL_PATH = 'bkt_rf_model.pkl' 
DATA_PATH = 'bkt_training_dataset_english_problem.csv'

# 세션 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.run_id = str(int(time.time())) # 브라우저 기만용 ID

# ===============================
# 2. 강력한 유틸리티 함수
# ===============================

def safe_translate(en_text):
    """번역 누락 및 f-string 구문 오류를 방지하는 강력한 번역 함수"""
    prompt = f"너는 수학 교사야. 다음 영문 문제를 반드시 한국어 '-하시오' 체로만 번역해. 영어는 절대 섞지 마. 문제: {en_text}"
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        ko_text = resp.text.strip()
        
        # AI가 번역을 거부하거나 영어를 그대로 뱉은 경우 대비
        if ko_text.lower() == en_text.lower() or len(re.findall(r'[a-zA-Z]', ko_text)) > len(ko_text) * 0.3:
            # f-string 내부 백슬래시 에러 방지를 위해 숫자 미리 추출
            num_match = re.search(r'\d+', en_text)
            extracted_num = num_match.group() if num_match else "제시된 수"
            return f"{extracted_num}을 소인수분해하시오."
        return ko_text
    except:
        return "문제를 번역하는 중 오류가 발생했습니다."

def diagnose_learning_status(results):
    """RF 모델 기반 진단 (variance 포함)"""
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
    except: return "진단 에러", 0, 0

# ===============================
# 3. 단계별 UI 로직
# ===============================

# [단계 0] 시작 화면
if st.session_state.step == 0:
    st.title("🎓 AI BKT 맞춤형 학습 시스템")
    st.write("학습을 시작하면 15문제를 한국어로 준비합니다.")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            # 이전 세션 찌꺼기 제거
            st.session_state.feedback_results = []
            st.session_state.new_recommendations = []
            
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            if len(pool) >= 15:
                selected_en = random.sample(pool, 15)
                temp_probs = []
                
                with st.spinner("문제를 한국어로 완벽하게 변환 중..."):
                    for i, en_text in enumerate(selected_en):
                        ko_text = safe_translate(en_text)
                        temp_probs.append({'id': i+1, 'question': ko_text, 'ans': ""})
                
                st.session_state.problems = temp_probs
                st.session_state.run_id = str(int(time.time())) # 시작할 때마다 ID 갱신
                st.session_state.step = 1
                st.rerun()
        else:
            st.error(f"파일을 찾을 수 없습니다: {DATA_PATH}")

# [단계 1] 1차 학습
elif st.session_state.step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{p.get('id', i+1)}.**")
        st.info(p.get('question'))
        
        # 💡 자동완성 차단 핵심: 라벨에 run_id를 섞어 브라우저를 속임
        dynamic_label = f"정답 입력 (ID:{st.session_state.run_id[-3:]}-{i})"
        p['ans'] = st.text_input(dynamic_label, key=f"ans_{i}_{st.session_state.run_id}")
        st.divider()

    if st.button("📤 답안 제출 및 분석", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

# [단계 2] 채점 및 AI 피드백
elif st.session_state.step == 2:
    st.title("🔍 상세 채점 결과")
    if not st.session_state.feedback_results:
        p_bar = st.progress(0)
        for i, p in enumerate(st.session_state.problems):
            prompt = (
                f"문제: {p.get('question')}\n학생 답: {p.get('ans')}\n"
                "지침: 1. 첫 줄은 O/X 2. 논리 분석 3. 보완 개념 4. ✅ 정답: [값]"
            )
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
            is_correct = 1 if resp.strip().startswith('O') else 0
            st.session_state.feedback_results.append({'id': p.get('id'), 'is_correct': is_correct, 'content': resp})
            
            if not is_correct:
                rec_q = client.models.generate_content(model=MODEL_NAME, contents=f"'{p.get('question')}'의 원리를 보완할 한글 문제를 하나 만들어줘.").text
                st.session_state.new_recommendations.append({'ref_id': p.get('id'), 'q': rec_q, 'ans': ""})
            p_bar.progress((i + 1) / 15)
        p_bar.empty()

    for res in st.session_state.feedback_results:
        if res.get('is_correct'): st.success(f"✅ Q{res.get('id')}\n{res.get('content')}")
        else: st.error(f"❌ Q{res.get('id')}\n{res.get('content')}")

    if st.session_state.new_recommendations:
        if st.button("🚀 부족한 부분 보완하기 (추천 문제)", type="primary"):
            st.session_state.step = 3
            st.rerun()
    else:
        st.balloons()
        if st.button("🔄 처음으로"): st.session_state.clear(); st.rerun()

# [단계 3] 추천 문제 풀이
elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec.get('q'))
        # 여기에도 자동완성 방지 적용
        rec['ans'] = st.text_input(f"추천 답 입력 (ID:{st.session_state.run_id[-2:]}-{i})", key=f"rec_{i}_{st.session_state.run_id}")
        st.divider()

    if st.button("🏁 최종 진단 결과 확인", type="primary"):
        st.session_state.step = 4
        st.rerun()

# [단계 4] 최종 진단
elif st.session_state.step == 4:
    st.title("🏆 최종 성취도 진단 리포트")
    status, acc, gain = diagnose_learning_status(st.session_state.feedback_results)
    
    report_prompt = f"상태:{status}, 정확도:{acc*100:.1f}%. 이 결과를 바탕으로 학생에게 학습 등급과 조언을 한국어로 해줘."
    report = client.models.generate_content(model=MODEL_NAME, contents=report_prompt).text
    
    st.success(f"### 분석 결과: {status}")
    st.markdown(report)
    
    if st.button("🔄 처음으로 돌아가기"):
        st.session_state.clear()
        st.rerun()
