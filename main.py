import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import joblib
import os
import time
import easyocr  # OCR 라이브러리 추가
from PIL import Image
from google import genai

# 1. 환경 설정
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

# OCR 모델 로드 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr_reader()

# API 설정 (Gemini 3 Flash 환경)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash" 

RF_MODEL_PATH = 'bkt_rf_model.pkl'
DATA_PATH = 'bkt_training_dataset_english_problem.csv'

# 세션 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.run_id = str(int(time.time() * 1000))

# ===============================
# 2. 핵심 로직 함수
# ===============================

def translate_problems_batch(en_list):
    """15문제를 한 번에 번역하여 차단 방지"""
    combined_query = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(en_list)])
    prompt = (
        "수학 교사로서 아래 영문 문제들을 순서대로 한국어 '-하시오' 체로 번역하시오.\n"
        "영어는 절대 섞지 말고 한국어 문장만 15개 나열하시오.\n\n" + combined_query
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        lines = resp.text.strip().split('\n')
        results = []
        for i, en_text in enumerate(en_list):
            try:
                line = [l for l in lines if str(i+1) in l[:5]][0]
                results.append(re.sub(r'^\d+[\.\s]*', '', line).strip())
            except:
                n = re.findall('[0-9]+', en_text)[0] if re.findall('[0-9]+', en_text) else "수"
                results.append(f"{n}을 소인수분해하시오.")
        return results
    except:
        return [f"{re.findall('[0-9]+', t)[0] if re.findall('[0-9]+', t) else '수'}를 소인수분해하시오." for t in en_list]

def diagnose_learning_status(results):
    """RF 모델 진단 연동"""
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
            if len(pool) >= 15:
                selected_en = random.sample(pool, 15)
                with st.spinner("AI 선생님이 문제를 준비 중입니다..."):
                    translated_ko = translate_problems_batch(selected_en)
                    # 데이터 구조에 'input_type'과 'img_ans' 추가
                    st.session_state.problems = [{'id': i+1, 'question': q, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"} for i, q in enumerate(translated_ko)]
                st.session_state.run_id = str(int(time.time() * 1000))
                st.session_state.step = 1
                st.rerun()

elif st.session_state.step == 1:
    st.title("📝 1차 학습")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p.get('question'))
        
        # 입력 방식 선택
        p['input_type'] = st.radio(f"답안 제출 방식 (Q{i+1})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"type_{i}_{st.session_state.run_id}", horizontal=True)

        if p['input_type'] == "⌨️ 타이핑":
            st.markdown('<input type="text" style="display:none;" name="fake_input">', unsafe_allow_html=True)
            r_str = "".join(random.choices("lmnop", k=8))
            p['ans'] = st.text_input(label=f"L_{r_str}_{i}", key=f"K_{st.session_state.run_id}_{i}_{r_str}", label_visibility="collapsed")
        else:
            # 카메라 입력 위젯
            captured_img = st.camera_input(f"Q{i+1} 풀이 촬영", key=f"cam_{i}_{st.session_state.run_id}")
            if captured_img:
                p['img_ans'] = captured_img
        st.divider()

    if st.button("📤 답안 제출", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 분석 결과")
    if not st.session_state.feedback_results:
        with st.spinner("AI가 채점 및 사진 분석 중..."):
            for p in st.session_state.problems:
                # 사진인 경우 OCR 실행
                final_ans = p['ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans'] is not None:
                    img = Image.open(p['img_ans'])
                    img_np = np.array(img)
                    ocr_result = reader.readtext(img_np, detail=0)
                    final_ans = " ".join(ocr_result)
                
                prompt = f"문제: {p['question']}\n학생 답: {final_ans}\nO/X 판정, 분석, 개념보강, 정답공개 순으로 작성."
                resp = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
                is_correct = 1 if resp.strip().startswith('O') else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'content': resp})
                if not is_correct:
                    rec_q = client.models.generate_content(model=MODEL_NAME, contents=f"'{p['question']}' 유사 한글 문제 생성.").text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'q': rec_q, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"})

    for res in st.session_state.feedback_results:
        if res['is_correct']: st.success(f"✅ Q{res['id']}\n{res['content']}")
        else: st.error(f"❌ Q{res['id']}\n{res['content']}")

    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제 풀기"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec.get('q'))
        rec['input_type'] = st.radio(f"방식 (추천 Q{i+1})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"rectype_{i}_{st.session_state.run_id}", horizontal=True)
        
        if rec['input_type'] == "⌨️ 타이핑":
            st.markdown('<input type="text" style="display:none;" name="fake_rec">', unsafe_allow_html=True)
            r_rec = "".join(random.choices("qrs", k=5))
            rec['ans'] = st.text_input(label=f"R_{r_rec}", key=f"RE_{i}_{r_rec}", label_visibility="collapsed")
        else:
            rec_img = st.camera_input(f"추천 Q{i+1} 촬영", key=f"reccam_{i}_{st.session_state.run_id}")
            if rec_img:
                rec['img_ans'] = rec_img
        st.divider()

    if st.button("🏁 최종 진단"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 최종 성취도")
    if not hasattr(st.session_state, 'bkt_report'):
        with st.spinner("최종 성취도 산출 중..."):
            # 추천 문제 OCR 처리 후 피드백 리스트 보강 (선택 사항)
            status, acc, gain = diagnose_learning_status(st.session_state.feedback_results)
            report = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정확도:{acc*100:.1f}%. 조언 한글로.").text
            st.session_state.bkt_report = report
            st.session_state.bkt_status = status

    st.success(f"### 성취 등급: {st.session_state.bkt_status}")
    st.markdown(st.session_state.bkt_report)
    if st.button("🔄 처음으로"):
        st.session_state.clear()
        st.rerun()
