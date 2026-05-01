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

# OCR 모델 로드 (캐싱)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr_reader()

# API 설정 (Gemini 3 Flash 환경)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
# 모델명을 현재 API에서 안정적으로 지원하는 명칭으로 유지합니다.
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
    """15문제를 묶어서 번역하며, 실패 시 숫자를 포함한 한글 문장 강제 생성"""
    combined_query = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(en_list)])
    prompt = (
        "수학 교사로서 아래 영문 문제들을 한국어 '-하시오' 체로 번역하시오.\n"
        "한국어 문장만 15개 나열하시오.\n\n" + combined_query
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
                nums = re.findall('[0-9]+', en_text)
                n = nums[0] if nums else "수"
                results.append(f"{n}을 소인수분해하시오.")
        return results
    except:
        fallback_results = []
        for t in en_list:
            nums = re.findall('[0-9]+', t)
            n = nums[0] if nums else "수"
            fallback_results.append(f"{n}를 소인수분해하시오.")
        return fallback_results

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
                    st.session_state.problems = [
                        {
                            'id': i+1, 
                            'question': q, 
                            'ans': "", 
                            'img_ans': None, 
                            'input_type': "⌨️ 타이핑"
                        } for i, q in enumerate(translated_ko)
                    ]
                st.session_state.step = 1
                st.rerun()

elif st.session_state.step == 1:
    st.title("📝 1차 학습")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p['question'])
        
        p['input_type'] = st.radio(f"제출 방식 (Q{i+1})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"type_{i}", horizontal=True)

        if p['input_type'] == "⌨️ 타이핑":
            p['ans'] = st.text_input(
                label="정답 입력", 
                value=p['ans'], 
                key=f"ans_input_{i}"
            )
        else:
            captured_img = st.camera_input(f"Q{i+1} 풀이 촬영", key=f"cam_{i}")
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
                final_ans = p['ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans'] is not None:
                    try:
                        img = Image.open(p['img_ans'])
                        img_np = np.array(img)
                        ocr_result = reader.readtext(img_np, detail=0)
                        final_ans = " ".join(ocr_result)
                    except:
                        final_ans = "(사진 분석 실패)"
                
                # 🛠️ [에러 수정 포인트] AI 호출 시 예외 처리 추가
                prompt = f"문제: {p['question']}\n학생 답: {final_ans}\nO/X 판정, 분석, 개념보강, 정답공개 순으로 작성."
                try:
                    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                    feedback_text = resp.text
                    is_correct = 1 if feedback_text.strip().startswith('O') else 0
                except Exception as e:
                    # AI 호출 실패 시 기본 메시지 출력
                    feedback_text = f"죄송합니다. AI 피드백을 생성하는 중 오류가 발생했습니다. (오류: {str(e)})"
                    is_correct = 0
                
                st.session_state.feedback_results.append({
                    'id': p['id'], 
                    'is_correct': is_correct, 
                    'content': feedback_text
                })
                
                # 오답인 경우 추천 문제 생성 (역시 예외 처리 적용)
                if is_correct == 0:
                    try:
                        rec_q_resp = client.models.generate_content(model=MODEL_NAME, contents=f"'{p['question']}' 유사 한글 문제 생성.")
                        rec_question = rec_q_resp.text
                    except:
                        rec_question = "유사 문제를 생성할 수 없습니다. 다시 시도해 주세요."
                    
                    st.session_state.new_recommendations.append({
                        'ref_id': p['id'], 'q': rec_question, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"
                    })

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
        st.info(rec['q'])
        rec['input_type'] = st.radio(f"방식 (추천 Q{i+1})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"rectype_{i}", horizontal=True)
        
        if rec['input_type'] == "⌨️ 타이핑":
            rec['ans'] = st.text_input("추천 문제 정답", value=rec['ans'], key=f"rec_ans_{i}")
        else:
            rec_img = st.camera_input(f"추천 Q{i+1} 촬영", key=f"reccam_{i}")
            if rec_img: rec['img_ans'] = rec_img
        st.divider()

    if st.button("🏁 최종 진단"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 최종 성취도")
    if 'bkt_report' not in st.session_state:
        status, acc, gain = diagnose_learning_status(st.session_state.feedback_results)
        # 최종 리포트 생성 시 예외 처리
        try:
            report_resp = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정확도:{acc*100:.1f}%. 조언 한글로.")
            st.session_state.bkt_report = report_resp.text
        except:
            st.session_state.bkt_report = "성취도 리포트를 생성하는 중 오류가 발생했습니다."
        st.session_state.bkt_status = status

    st.success(f"### 성취 등급: {st.session_state.bkt_status}")
    st.markdown(st.session_state.bkt_report)
    if st.button("🔄 처음으로"):
        st.session_state.clear()
        st.rerun()
