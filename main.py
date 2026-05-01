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

# OCR 모델 로드 (캐싱)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr()

# API 설정 (Streamlit Cloud Secrets에 GEMINI_API_KEY 입력 필수)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# 파일 경로 (GitHub Root 기준)
RF_MODEL_PATH = 'bkt_rf_model.pkl' 
DATA_PATH = 'bkt_training_dataset_english_problem.csv'

# 세션 상태 초기화 (데이터 구조 변경 시 충돌 방지를 위해 run_id 기반 관리)
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.final_feedback = []
    st.session_state.run_id = str(time.time()).replace(".", "")[-6:]

# ===============================
# 2. 핵심 로직 함수
# ===============================

def analyze_answer_logic(question, student_ans):
    """1, 2단계 공통 상세 채점 로직"""
    prompt = (
        f"문제: {question}\n학생의 답: {student_ans}\n\n"
        "지침:\n"
        "1. 정오답 여부를 첫 줄에 'O' 또는 'X'로 표시할 것.\n"
        "2. 학생의 풀이 과정에서 나타난 논리적 오류나 식의 잘못된 부분을 구체적으로 분석할 것.\n"
        "3. 이 문제를 풀기 위해 보완해야 할 수학적 개념을 설명할 것.\n"
        "4. ✅ 정답: [최종 결과값]을 반드시 포함하여 알려줄 것."
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return resp.text
    except Exception as e:
        return f"X\n분석 중 오류가 발생했습니다: {str(e)}"

def diagnose_learning_status(results):
    """RF 모델 기반 BKT 진단 (variance 포함)"""
    if not os.path.exists(RF_MODEL_PATH):
        return "MODEL_NOT_FOUND", 0, 0, 0
    try:
        model = joblib.load(RF_MODEL_PATH)
        correctness = [r.get('is_correct', 0) for r in results]
        accuracy = sum(correctness) / len(results) if results else 0
        initial_k = correctness[0] * 0.4 if correctness else 0
        final_k = correctness[-1] * 0.8 if correctness else 0
        learning_gain = max(0, final_k - initial_k)
        variance = np.var(correctness) if len(correctness) > 1 else 0.0

        input_df = pd.DataFrame([[initial_k, final_k, learning_gain, accuracy, variance]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        status = model.predict(input_df)[0]
        return status, accuracy, learning_gain, initial_k
    except:
        return "DIAGNOSIS_ERROR", 0, 0, 0

def ocr_process(image_file):
    try:
        img = Image.open(image_file)
        img_np = np.array(img)
        res = reader.readtext(img_np)
        return " ".join([r[1] for r in res])
    except:
        return "이미지 인식 실패"

# ===============================
# 3. 단계별 UI 로직
# ===============================

# [단계 0] 시작 화면
if st.session_state.step == 0:
    st.title("🎓 AI BKT 맞춤형 학습 시스템")
    st.write("LG-AI-CAMP-NEW 데이터셋에서 오늘의 문제를 준비합니다.")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            if len(pool) >= 15:
                # 새로운 run_id 생성 (자동완성 방지용)
                st.session_state.run_id = str(time.time()).replace(".", "")[-6:]
                
                selected_en = random.sample(pool, 15)
                temp_probs = []
                # 번역 과정을 화면에 표시하지 않고 백그라운드 처리
                for i, en_text in enumerate(selected_en):
                    try:
                        ko_text = client.models.generate_content(
                            model=MODEL_NAME, 
                            contents=f"Translate this math problem to Korean only, using formal '-하시오' style: {en_text}"
                        ).text
                    except:
                        ko_text = en_text
                    temp_probs.append({
                        'id': i+1, 
                        'question': ko_text, 
                        'text_ans': "", 
                        'img_ans': None, 
                        'input_type': '⌨️ 타이핑'
                    })
                
                st.session_state.problems = temp_probs
                st.session_state.feedback_results = []
                st.session_state.new_recommendations = []
                st.session_state.final_feedback = []
                st.session_state.step = 1
                st.rerun()
        else:
            st.error("데이터셋 파일을 찾을 수 없습니다. (bkt_training_dataset_english_problem.csv)")

# [단계 1] 1차 학습 (15문제)
elif st.session_state.step == 1:
    st.title("📝 1차 학습: 오늘의 15문제")
    
    # KeyError 방지를 위해 enumerate와 .get() 사용
    for i, p in enumerate(st.session_state.problems):
        with st.container():
            q_id = p.get('id', i+1)
            q_text = p.get('question', '문제를 불러오는 중입니다...')
            
            st.markdown(f"### **Q{q_id}.**")
            st.info(q_text)
            
            # 자동 완성 팝업 차단을 위해 key와 label에 run_id 조합
            p['input_type'] = st.radio(f"답변 방식 (Q{q_id})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"r_{i}_{st.session_state.run_id}", horizontal=True)
            
            if p['input_type'] == "⌨️ 타이핑":
                # 라벨에 run_id를 포함하여 브라우저의 자동완성 로직을 무력화
                p['text_ans'] = st.text_input(f"정답 입력 (코드:{st.session_state.run_id})", key=f"in_{i}_{st.session_state.run_id}")
            else:
                img = st.camera_input(f"Capture (Q{q_id})", key=f"cam_{i}_{st.session_state.run_id}")
                if img: p['img_ans'] = img
            st.divider()

    if st.button("📤 답안 제출 및 분석 시작", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

# [단계 2] 1차 채점 분석
elif st.session_state.step == 2:
    st.title("🔍 1차 채점 결과 및 상세 분석")
    
    if not st.session_state.feedback_results:
        status_bar = st.progress(0)
        for i, p in enumerate(st.session_state.problems):
            # 답안 추출
            ans_text = p.get('text_ans', "")
            if p.get('input_type') == "📸 사진 찍기" and p.get('img_ans'):
                ans_text = ocr_process(p['img_ans'])
            
            # 상세 분석 실행
            res_text = analyze_answer_logic(p.get('question', ''), ans_text)
            is_correct = 1 if res_text.strip().startswith('O') else 0
            
            st.session_state.feedback_results.append({
                'id': p.get('id', i+1), 
                'is_correct': is_correct, 
                'content': res_text
            })
            
            # 오답 시 추천 문제 생성
            if is_correct == 0:
                try:
                    gen_prompt = f"이 문제('{p.get('question', '')}')와 비슷한 원리의 한글 변형 문제를 하나 만들어줘."
                    new_q = client.models.generate_content(model=MODEL_NAME, contents=gen_prompt).text
                    st.session_state.new_recommendations.append({'ref_id': p.get('id', i+1), 'new_question': new_q, 'new_ans': ""})
                except:
                    pass
            
            status_bar.progress((i + 1) / len(st.session_state.problems))
        status_bar.empty()

    # 결과 출력
    for res in st.session_state.feedback_results:
        if res.get('is_correct'):
            st.success(f"✅ **Q{res.get('id')}**\n\n{res.get('content')}")
        else:
            st.error(f"❌ **Q{res.get('id')} 오답 분석**\n\n{res.get('content')}")
        st.divider()
    
    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제 풀기 (2단계)", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    else:
        st.balloons()
        if st.button("🔄 처음으로"): 
            st.session_state.clear()
            st.rerun()

# [단계 3] 2차 추천 문제 풀이
elif st.session_state.step == 3:
    st.title("🎯 2차 학습: 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.markdown(f"### **추천 문제 (Q{rec.get('ref_id')} 보완)**")
        st.info(rec.get('new_question', '문제를 로드하지 못했습니다.'))
        rec['new_ans'] = st.text_input(f"답 입력 (코드:{st.session_state.run_id})", key=f"f_{i}_{st.session_state.run_id}")
        st.divider()

    if st.button("🏁 최종 제출 및 진단", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

# [단계 4] 최종 결과 및 BKT 진단
elif st.session_state.step == 4:
    st.title("🏆 최종 성취도 진단 결과")
    
    if not st.session_state.final_feedback:
        with st.spinner("최종 성취도를 분석 중입니다..."):
            for rec in st.session_state.new_recommendations:
                # 추천 문제도 동일한 깊이로 분석
                res_text = analyze_answer_logic(rec.get('new_question', ''), rec.get('new_ans', ''))
                st.session_state.final_feedback.append({'ref_id': rec.get('ref_id'), 'content': res_text})
            
            # RF 모델용 데이터 결합
            c1 = [r.get('is_correct', 0) for r in st.session_state.feedback_results]
            c2 = [1 if f.get('content', '').strip().startswith('O') else 0 for f in st.session_state.final_feedback]
            status, acc, gain, init = diagnose_learning_status(st.session_state.feedback_results + [{'is_correct': v} for v in c2])
            
            summary_prompt = f"상태:{status}, 정확도:{acc*100:.1f}%, 성장도:{gain:.2f}. 위 데이터를 근거로 학생의 학습 등급(A~E)과 최종 조언을 한글로 작성해줘."
            st.session_state.bkt_report = client.models.generate_content(model=MODEL_NAME, contents=summary_prompt).text

    for res in st.session_state.final_feedback:
        st.info(f"📝 **추천 문제 분석 (Q{res.get('ref_id')} 관련)**\n\n{res.get('content')}")
    
    st.success("### 📊 종합 BKT 성취 리포트")
    st.markdown(st.session_state.bkt_report)
    st.balloons()
    
    if st.button("🔄 처음으로 돌아가기"):
        st.session_state.clear()
        st.rerun()
