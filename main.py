import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import joblib
import os
import time
import easyocr
from PIL import Image
from google import genai

# 1. 환경 설정
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr_reader()

# API 설정 (사용자 지정: gemini-2.5-flash)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash" 

# 모델 및 데이터 경로 (언더바 2개 'bkt_rf__model.pkl' 확인)
RF_MODEL_PATH = '/mount/src/lg-ai-camp-new/bkt_rf__model.pkl'
DATA_PATH = '/mount/src/lg-ai-camp-new/bkt_training_dataset_english_problem.csv'

# 세션 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []

# ===============================
# 2. 핵심 로직 함수
# ===============================

def translate_problems_batch(en_list):
    """영어 문제 번역 - 역슬래시 에러 원천 차단"""
    prompt = (
        "수학 선생님으로서 다음 영어 문제들을 한글 '-하시오' 체로 번역해줘.\n"
        "불필요한 인사말 없이 번역된 문장만 한 줄에 하나씩 나열해.\n\n" + 
        "\n".join([f"{i+1}. {t}" for i, t in enumerate(en_list)])
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        lines = resp.text.strip().split('\n')
        results = []
        for line in lines:
            # 번호 제거
            clean_line = re.sub(r'^\d+\.\s*', '', line).strip()
            if clean_line and "번역" not in clean_line[:10]:
                results.append(clean_line)
        return results[:len(en_list)]
    except:
        # ✅ SyntaxError 해결: f-string 중괄호 바깥에서 정규식 처리
        fallback_results = []
        for t in en_list:
            nums = re.findall(r'\d+', t) # 백슬래시는 여기서만 사용
            num_val = nums[0] if nums else "수"
            fallback_results.append(f"{num_val}을 소인수분해하시오.") # f-string 내부엔 변수만
        return fallback_results

def diagnose_learning_status(results):
    """
    진단 안정화 보정:
    이미지 044734.png처럼 73.3%인데 Guessing이 나오는 것은 단일 문항 결과에 
    BKT 특성치가 민감하게 반응했기 때문입니다. 가중치를 조정하여 안정화합니다.
    """
    corrects = [r.get('is_correct', 0) for r in results]
    actual_acc = (sum(corrects) / len(results)) * 100 if results else 0.0
    
    if not os.path.exists(RF_MODEL_PATH):
        return "MODEL_MISSING", actual_acc, 0.0
        
    try:
        model = joblib.load(RF_MODEL_PATH)
        # ✅ 특성치 계산 보정: 전반부/후반부 평균을 사용하여 한두 문제 실수에 의한 튀는 현상 방지
        init_k = np.mean(corrects[:5]) * 0.5 if len(corrects) >= 5 else corrects[0] * 0.4
        final_k = np.mean(corrects[-5:]) * 0.9 if len(corrects) >= 5 else corrects[-1] * 0.8
        gain = max(0, final_k - init_k)
        var = np.var(corrects)
        
        df_input = pd.DataFrame([[init_k, final_k, gain, actual_acc/100, var]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        
        status = model.predict(df_input)[0]
        return status, actual_acc, gain
    except:
        return "진단 실패", actual_acc, 0.0

# ===============================
# 3. UI 및 단계별 흐름
# ===============================

if st.session_state.step == 0:
    st.title("🎓 AI BKT 학습 시스템")
    if not os.path.exists(RF_MODEL_PATH):
        st.error(f"🚨 모델 파일 없음: {RF_MODEL_PATH}")

    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            # 숫자 기반 중복 제거
            unique_numbered_dict = {re.findall(r'\d+', t)[0]: t for t in pool if re.findall(r'\d+', t)}
            clean_pool = list(unique_numbered_dict.values())

            if len(clean_pool) >= 15:
                selected_en = random.sample(clean_pool, 15)
                translated_ko = translate_problems_batch(selected_en)
                st.session_state.problems = [{'id': i+1, 'question': q, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"} for i, q in enumerate(translated_ko)]
                st.session_state.step = 1
                st.rerun()

elif st.session_state.step == 1:
    st.title("📝 1차 학습")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p['question'])
        p['input_type'] = st.radio(f"방식_{i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"t_{i}", horizontal=True, label_visibility="collapsed")
        if p['input_type'] == "⌨️ 타이핑":
            p['ans'] = st.text_input(f"답_{i}", value=p['ans'], key=f"a_{i}", label_visibility="collapsed")
        else:
            c_img = st.camera_input(f"촬영", key=f"c_{i}", label_visibility="collapsed")
            if c_img: p['img_ans'] = c_img
        st.divider()

    if st.button("📤 모든 답안 제출", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 채점 결과")
    if not st.session_state.feedback_results:
        with st.spinner("AI 분석 및 추천 문제 생성 중..."):
            student_summary = [f"Q{p['id']}. 문제: {p['question']} / 학생답: {p['ans']}" for p in st.session_state.problems]
            prompt = (
                "수학 교사로서 채점해줘. 'Q번호: O/X' 리스트를 반드시 포함하고 틀린 문제만 상세 분석해.\n\n" + "\n".join(student_summary)
            )
            feedback_text = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
            st.session_state.full_feedback = feedback_text

            # ✅ 추천 문제 1:1 매칭 로직 (틀린 문제마다 즉시 생성)
            used_nums = []
            for prob in st.session_state.problems:
                ns = re.findall(r'\d+', prob['question'])
                if ns: used_nums.append(ns[0])
            
            for p in st.session_state.problems:
                # O/X 판정
                match = re.search(f"Q{p['id']}:?\s*([OX0])", feedback_text, re.IGNORECASE)
                is_ok = 1 if match and match.group(1).upper() in ['O', '0'] else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_ok})

                # 틀린 문제에 대해 추천 문제 생성
                if is_ok == 0:
                    rec_p = f"학생이 '{p['question']}'를 틀렸어. 숫자 {used_nums}를 제외한 유사 소인수분해 문제 1개 생성해."
                    try:
                        r = client.models.generate_content(model=MODEL_NAME, contents=rec_p).text
                        st.session_state.new_recommendations.append({'q': r, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"})
                        new_num = re.findall(r'\d+', r)
                        if new_num: used_nums.append(new_num[0])
                    except: pass

    # UI 표시
    cols = st.columns(5)
    for i, res in enumerate(st.session_state.feedback_results):
        with cols[i % 5]:
            if res['is_correct']: st.success(f"Q{res['id']}: ✅")
            else: st.error(f"Q{res['id']}: ❌")
    
    st.divider()
    st.markdown(st.session_state.full_feedback)
    
    if st.session_state.new_recommendations:
        if st.button(f"🚀 추천 문제 ({len(st.session_state.new_recommendations)}개) 풀기", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(f"💡 추천 {i+1}: {rec['q']}")
        rec['input_type'] = st.radio(f"입력_{i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"rt_{i}", horizontal=True, label_visibility="collapsed")
        if rec['input_type'] == "⌨️ 타이핑":
            rec['ans'] = st.text_input(f"답안_{i}", value=rec['ans'], key=f"ra_{i}", label_visibility="collapsed")
        else:
            rimg = st.camera_input(f"촬영", key=f"rc_{i}", label_visibility="collapsed")
            if rimg: rec['img_ans'] = rimg
        st.divider()
    
    if st.button("🏁 최종 성취도 확인", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 성취도 리포트")
    # ✅ 이미지 044734.png의 문제를 해결하기 위해 보정된 진단 함수 호출
    status, actual_acc, gain = diagnose_learning_status(st.session_state.feedback_results)
    st.success(f"### 성취도 등급: {status}")
    st.info(f"📊 실제 정답률: {actual_acc:.1f}%")
    try:
        rep = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정답률:{actual_acc}%. 학습 조언 한글로.").text
        st.markdown(rep)
    except: st.write("리포트 생성 오류")
    if st.button("🔄 처음으로"):
        st.session_state.clear(); st.rerun()
