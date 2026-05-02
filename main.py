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

# API 설정
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash" 

RF_MODEL_PATH = 'bkt_rf_model.pkl'
DATA_PATH = 'bkt_training_dataset_english_problem.csv'

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
    """영어 문제를 한글로 번역"""
    prompt = (
        "수학 선생님으로서 다음 영어 문제들을 한글 '-하시오' 체로 번역해줘.\n"
        "각 줄에 하나씩 번역된 문장만 나열해.\n\n" + 
        "\n".join([f"{i+1}. {t}" for i, t in enumerate(en_list)])
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return [line.split('. ')[-1].strip() for line in resp.text.strip().split('\n') if line]
    except:
        # ✅ [SyntaxError 해결] f-string 내부에서 직접 re.findall을 호출하지 않고 외부에서 변수화
        fallback_results = []
        for t in en_list:
            nums = re.findall(r'\d+', t)
            num_val = nums[0] if nums else "수"
            fallback_results.append(f"{num_val}를 소인수분해하시오.")
        return fallback_results

def diagnose_learning_status(results):
    if not os.path.exists(RF_MODEL_PATH): return "MODEL_MISSING", 0, 0
    try:
        model = joblib.load(RF_MODEL_PATH)
        corrects = [r.get('is_correct', 0) for r in results]
        acc = sum(corrects) / len(results) if results else 0
        init_k, final_k = corrects[0]*0.4, corrects[-1]*0.8
        var = np.var(corrects) if len(corrects) > 1 else 0.0
        df = pd.DataFrame([[init_k, final_k, max(0, final_k-init_k), acc, var]], 
                          columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        return model.predict(df)[0], acc, max(0, final_k-init_k)
    except: return "ERROR", 0, 0

# ===============================
# 3. UI 및 단계별 흐름
# ===============================

if st.session_state.step == 0:
    st.title("🎓 AI BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            
            # ✅ [숫자 기반 중복 제거]
            unique_numbered_dict = {}
            for text in pool:
                nums = re.findall(r'\d+', text)
                if nums:
                    num_val = nums[0]
                    if num_val not in unique_numbered_dict:
                        unique_numbered_dict[num_val] = text
            
            clean_pool = list(unique_numbered_dict.values())

            if len(clean_pool) >= 15:
                selected_en = random.sample(clean_pool, 15)
                with st.spinner("숫자 중복 검사 완료! 번역 중..."):
                    translated_ko = translate_problems_batch(selected_en)
                    st.session_state.problems = [
                        {'id': i+1, 'question': q, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"} 
                        for i, q in enumerate(translated_ko)
                    ]
                st.session_state.step = 1
                st.rerun()

elif st.session_state.step == 1:
    st.title("📝 1차 학습 (15문항)")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p['question'])
        p['input_type'] = st.radio(f"방식_{i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"t_{i}", horizontal=True, label_visibility="collapsed")
        
        if p['input_type'] == "⌨️ 타이핑":
            p['ans'] = st.text_input(f"답_{i}", value=p['ans'], key=f"a_{i}", label_visibility="collapsed", placeholder="정답 입력")
        else:
            c_img = st.camera_input(f"카메라_{i}", key=f"c_{i}", label_visibility="collapsed")
            if c_img: p['img_ans'] = c_img
        st.divider()

    if st.button("📤 모든 답안 제출", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 채점 결과")
    if not st.session_state.feedback_results:
        with st.spinner("채점 및 분석 중..."):
            student_summary = []
            for p in st.session_state.problems:
                f_ans = p['ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans'] is not None:
                    try:
                        img = Image.open(p['img_ans'])
                        ocr_res = reader.readtext(np.array(img), detail=0)
                        f_ans = " ".join(ocr_res)
                    except: f_ans = "(OCR 실패)"
                student_summary.append(f"Q{p['id']}. 문제: {p['question']} / 학생답: {f_ans}")

            prompt = (
                "수학 선생님으로서 채점해줘.\n"
                "1. [채점 요약]: 모든 문항에 대해 'Q1: O', 'Q2: X' 형식으로 리스트를 작성해.\n"
                "2. [오답 클리닉]: 틀린 문제(X)에 대해서만 문제 원문, 상세 분석, 정답을 길게 설명해.\n"
                "맞은 문제는 절대 설명하지 마.\n\n" + "\n".join(student_summary)
            )
            
            try:
                feedback_text = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
            except:
                feedback_text = "피드백 생성 오류가 발생했습니다."

            for p in st.session_state.problems:
                is_ok = 1 if f"Q{p['id']}: O" in feedback_text or f"Q{p['id']}:O" in feedback_text else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_ok})
            st.session_state.full_feedback = feedback_text

    # --- UI 표시 ---
    st.subheader("📊 요약 리스트")
    cols = st.columns(5)
    for i, res in enumerate(st.session_state.feedback_results):
        with cols[i % 5]:
            if res['is_correct']: st.success(f"Q{res['id']}: ✅")
            else: st.error(f"Q{res['id']}: ❌")
    
    st.divider()
    st.subheader("💡 오답 상세 분석 (틀린 문제만)")
    st.markdown(st.session_state.full_feedback)
    
    if st.button("🚀 추천 문제로 보완하기", type="primary", use_container_width=True):
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    used_nums = []
    for p in st.session_state.problems:
        ns = re.findall(r'\d+', p['question'])
        if ns: used_nums.append(ns[0])
    
    if not st.session_state.new_recommendations:
        for res in st.session_state.feedback_results:
            if not res['is_correct']:
                orig_q = st.session_state.problems[res['id']-1]['question']
                rec_p = f"'{orig_q}' 오답 보완용. 숫자 {used_nums}를 제외한 새로운 유사 한글 문제 1개 생성."
                try:
                    r = client.models.generate_content(model=MODEL_NAME, contents=rec_p).text
                    st.session_state.new_recommendations.append({'q': r, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"})
                except: pass

    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec['q'])
        rec['input_type'] = st.radio(f"입력_{i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"rt_{i}", horizontal=True, label_visibility="collapsed")
        if rec['input_type'] == "⌨️ 타이핑":
            rec['ans'] = st.text_input(f"답안_{i}", value=rec['ans'], key=f"ra_{i}", label_visibility="collapsed")
        else:
            rimg = st.camera_input(f"카메라추천_{i}", key=f"rc_{i}", label_visibility="collapsed")
            if rimg: rec['img_ans'] = rimg
        st.divider()
    
    if st.button("🏁 최종 결과 확인", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 성취도 리포트")
    status, acc, _ = diagnose_learning_status(st.session_state.feedback_results)
    st.success(f"### 성취도 등급: {status}")
    try:
        rep = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정확도:{acc*100:.1f}%. 조언 한글로.").text
        st.markdown(rep)
    except: st.write("리포트 생성 오류")
    if st.button("🔄 처음으로"):
        st.session_state.clear(); st.rerun()
