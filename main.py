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

# API 설정 (Gemini 2.5 Flash)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash" 

# ✅ [경로 해결 핵심] 하드코딩 대신 현재 파일 위치를 기준으로 경로를 생성합니다.
# 이렇게 하면 로컬 환경이든 배포 환경(/mount/src/...)이든 상관없이 파일을 찾아냅니다.
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(CUR_DIR, 'bkt_rf_model.pkl')
DATA_PATH = os.path.join(CUR_DIR, 'bkt_training_dataset_english_problem.csv')

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
    """영어 문제 번역 (인사말 차단)"""
    prompt = (
        "수학 선생님으로서 다음 영어 문제들을 한글 '-하시오' 체로 번역해줘.\n"
        "주의: '다음은 번역입니다' 같은 군더더기는 절대 하지 말고 번역된 문장만 나열해.\n"
        "각 문장 앞에 '1.', '2.' 처럼 반드시 번호를 붙여서 출력해.\n\n" + 
        "\n".join([f"{i+1}. {t}" for i, t in enumerate(en_list)])
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        lines = resp.text.strip().split('\n')
        results = []
        for line in lines:
            clean_line = re.sub(r'^\d+\.\s*', '', line).strip()
            if clean_line and "번역" not in clean_line[:10]:
                results.append(clean_line)
        if len(results) < len(en_list):
            for i in range(len(results), len(en_list)):
                nums = re.findall(r'\d+', en_list[i])
                num_val = nums[0] if nums else "수"
                results.append(f"{num_val}을 소인수분해하시오.")
        return results[:len(en_list)]
    except:
        return [f"{re.findall(r'\d+', t)[0] if re.findall(r'\d+', t) else '수'}을 소인수분해하시오." for t in en_list]

def diagnose_learning_status(results):
    """모델 로드 및 진단"""
    corrects = [r.get('is_correct', 0) for r in results]
    actual_acc = (sum(corrects) / len(results)) * 100 if results else 0.0
    
    # ✅ 파일 존재 여부 실시간 체크
    if not os.path.exists(RF_MODEL_PATH):
        return f"MODEL_MISSING: {RF_MODEL_PATH}", actual_acc, 0.0
        
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
        return f"ERROR: {str(e)}", actual_acc, 0.0

# ===============================
# 3. UI 및 단계별 흐름
# ===============================

if st.session_state.step == 0:
    st.title("🎓 AI BKT 학습 시스템")
    
    # ✅ 배포 전 상태 체크
    if not os.path.exists(RF_MODEL_PATH):
        st.error(f"🚨 모델 파일이 없습니다! 깃허브에 'bkt_rf_model.pkl'이 있는지 확인해주세요.")
        st.code(f"현재 경로: {CUR_DIR}\n필요 파일: {RF_MODEL_PATH}")

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
    st.title("📝 1차 학습")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p['question'])
        p['input_type'] = st.radio(f"방식_{i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"t_{i}", horizontal=True, label_visibility="collapsed")
        if p['input_type'] == "⌨️ 타이핑":
            p['ans'] = st.text_input(f"답_{i}", value=p['ans'], key=f"a_{i}", label_visibility="collapsed", placeholder="정답 입력")
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
        with st.spinner("AI 분석 중..."):
            student_summary = [f"Q{p['id']}. 문제: {p['question']} / 학생답: {p['ans']}" for p in st.session_state.problems]
            prompt = (
                "수학 교사로서 채점해줘.\n"
                "1. [채점 리스트]: 'Q1: O', 'Q2: X' 형식으로 작성.\n"
                "2. [상세 분석]: 틀린 문제(X)만 상세 분석.\n\n" + "\n".join(student_summary)
            )
            feedback_text = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
            
            for p in st.session_state.problems:
                is_ok = 1 if f"Q{p['id']}: O" in feedback_text or f"Q{p['id']}:O" in feedback_text else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_ok})
            st.session_state.full_feedback = feedback_text

    st.subheader("📊 요약 리스트")
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
            used_nums = [re.findall(r'\d+', p['question'])[0] for p in st.session_state.problems if re.findall(r'\d+', p['question'])]
            for res in st.session_state.feedback_results:
                if not res['is_correct']:
                    orig_q = st.session_state.problems[res['id']-1]['question']
                    rec_p = f"학생이 '{orig_q}'를 틀렸어. 숫자 {used_nums}를 제외한 유사 소인수분해 문제 1개 생성."
                    try:
                        r = client.models.generate_content(model=MODEL_NAME, contents=rec_p).text
                        st.session_state.new_recommendations.append({'q': r, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"})
                    except: pass

    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(f"💡 추천 문제 {i+1}")
        st.write(rec['q'])
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
