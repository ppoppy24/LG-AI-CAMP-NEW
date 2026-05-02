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

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, 'bkt_rf_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'bkt_training_dataset_english_problem.csv')

# 세션 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []

# ===============================
# 2. 핵심 로직 함수 (수정됨)
# ===============================

def translate_problems_batch(en_list):
    """영어 문제를 한글로 번역 - 인사말 차단 로직 강화"""
    prompt = (
        "수학 선생님으로서 다음 영어 문제들을 한글 '-하시오' 체로 번역해줘.\n"
        "⚠️주의: '다음은 번역입니다' 같은 인사말이나 서론은 절대 포함하지 마.\n" # 인사말 차단 프롬프트
        "오직 번역된 문제 문장만 각 줄에 하나씩 나열해.\n\n" + 
        "\n".join([f"{i+1}. {t}" for i, t in enumerate(en_list)])
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        lines = resp.text.strip().split('\n')
        
        results = []
        for line in lines:
            # 1. 숫자로 시작하는 줄에서 내용만 추출 (예: "1. 20을~" -> "20을~")
            clean_line = re.sub(r'^\d+[\.\s]*', '', line).strip()
            # 2. 내용이 비어있지 않고, '번역'이라는 단어가 포함된 인사말이 아닌 경우만 추가
            if clean_line and "번역" not in clean_line[:10]:
                results.append(clean_line)
        
        # 개수가 부족하면 숫자를 추출해 강제 생성
        if len(results) < len(en_list):
            for i in range(len(results), len(en_list)):
                nums = re.findall(r'\d+', en_list[i])
                num_val = nums[0] if nums else "수"
                results.append(f"{num_val}를 소인수분해하시오.")
                
        return results[:len(en_list)]
    except:
        fallback_results = []
        for t in en_list:
            nums = re.findall(r'\d+', t)
            num_val = nums[0] if nums else "수"
            fallback_results.append(f"{num_val}를 소인수분해하시오.")
        return fallback_results

def diagnose_learning_status(results):
    corrects = [r.get('is_correct', 0) for r in results]
    actual_acc = (sum(corrects) / len(results)) * 100 if results else 0.0
    
    if not os.path.exists(RF_MODEL_PATH):
        return "MODEL_MISSING (파일 없음)", actual_acc, 0.0
        
    try:
        model = joblib.load(RF_MODEL_PATH)
        init_k = corrects[0] * 0.4
        final_k = corrects[-1] * 0.8
        gain = max(0, final_k - init_k)
        var = np.var(corrects) if len(corrects) > 1 else 0.0
        
        df_input = pd.DataFrame([[init_k, final_k, gain, actual_acc/100, var]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        
        status = model.predict(df_input)[0]
        return status, actual_acc, gain
    except Exception as e:
        return f"ERROR: {type(e).__name__}", actual_acc, 0.0

# ===============================
# 3. UI 및 단계별 흐름
# ===============================

if st.session_state.step == 0:
    st.title("🎓 AI BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            
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
                with st.spinner("문제를 준비 중입니다..."):
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
        with st.spinner("AI 분석 중..."):
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
                "수학 교사로서 채점해줘.\n"
                "1. [채점 리스트]: 모든 문항에 대해 'Q1: O', 'Q2: X' 형식으로 작성해.\n"
                "2. [상세 분석]: 틀린 문제(X)만 골라서 아주 상세한 풀이와 정답을 작성해.\n"
                "맞은 문제는 절대 설명하지 마.\n\n" + "\n".join(student_summary)
            )
            
            feedback_text = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
            
            for p in st.session_state.problems:
                is_ok = 1 if f"Q{p['id']}: O" in feedback_text or f"Q{p['id']}:O" in feedback_text else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_ok})
            st.session_state.full_feedback = feedback_text

    st.subheader("📊 요약 결과")
    cols = st.columns(5)
    for i, res in enumerate(st.session_state.feedback_results):
        with cols[i % 5]:
            if res['is_correct']: st.success(f"Q{res['id']}: ✅")
            else: st.error(f"Q{res['id']}: ❌")
    
    st.divider()
    st.markdown(st.session_state.full_feedback)
    
    if st.button("🚀 추천 문제로 보완하기", type="primary", use_container_width=True):
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    if not st.session_state.new_recommendations:
        with st.spinner("추천 문제 생성 중..."):
            used_nums = []
            for p in st.session_state.problems:
                ns = re.findall(r'\d+', p['question'])
                if ns: used_nums.append(ns[0])

            for res in st.session_state.feedback_results:
                if not res['is_correct']:
                    orig_q = st.session_state.problems[res['id']-1]['question']
                    rec_p = f"학생이 '{orig_q}'를 틀렸어. 숫자 {used_nums}를 제외한 유사 소인수분해 문제 1개 생성."
                    try:
                        r = client.models.generate_content(model=MODEL_NAME, contents=rec_p).text
                        st.session_state.new_recommendations.append({'q': r, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"})
                        new_ns = re.findall(r'\d+', r)
                        if new_ns: used_nums.append(new_ns[0])
                    except: pass

    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(f"💡 추천 문제 {i+1}")
        st.write(rec['q'])
        rec['input_type'] = st.radio(f"입력_{i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"rt_{i}", horizontal=True, label_visibility="collapsed")
        if rec['input_type'] == "⌨️ 타이핑":
            rec['ans'] = st.text_input(f"답안_{i}", value=rec['ans'], key=f"ra_{i}", label_visibility="collapsed")
        else:
            rimg = st.camera_input(f"카메라추천_{i}", key=f"rc_{i}", label_visibility="collapsed")
            if rimg: rec['img_ans'] = rimg
        st.divider()
    
    if st.button("🏁 최종 성취도 확인", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 성취도 리포트")
    status, actual_acc, gain = diagnose_learning_status(st.session_state.feedback_results)
    st.success(f"### 성취도 등급: {status}")
    st.info(f"📊 실제 정답률: {actual_acc:.1f}%")
    
    try:
        rep = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 실제정답률:{actual_acc}%. 학습 조언 한글로.").text
        st.markdown(rep)
    except:
        st.write("리포트 생성 오류")
        
    if st.button("🔄 처음으로 돌아가기"):
        st.session_state.clear(); st.rerun()
