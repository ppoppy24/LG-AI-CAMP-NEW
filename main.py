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

# API 설정 (사용자 지정 모델: gemini-2.5-flash)
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
                n = re.findall('[0-9]+', en_text)[0] if re.findall('[0-9]+', en_text) else "수"
                results.append(f"{n}을 소인수분해하시오.")
        return results
    except:
        return [f"{re.findall('[0-9]+', t)[0] if re.findall('[0-9]+', t) else '수'}를 소인수분해하시오." for t in en_list]

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
            
            # ✅ [중복 차단 강화] 숫자 기반 엄격 필터링
            pool = df['generated_problem_english'].dropna().unique().tolist()
            unique_numbered_pool = {}
            for text in pool:
                nums = re.findall(r'\d+', text)
                num_key = nums[0] if nums else text
                if num_key not in unique_numbered_pool:
                    unique_numbered_pool[num_key] = text
            
            final_pool = list(unique_numbered_pool.values())

            if len(final_pool) >= 15:
                selected_en = random.sample(final_pool, 15)
                with st.spinner("중복 검사를 마치고 문제를 가져오고 있습니다..."):
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
        p['input_type'] = st.radio(f"제출 방식 (Q{i+1})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"type_{i}", horizontal=True)
        if p['input_type'] == "⌨️ 타이핑":
            p['ans'] = st.text_input(label="정답 입력", value=p['ans'], key=f"ans_input_{i}")
        else:
            captured_img = st.camera_input(f"Q{i+1} 촬영", key=f"cam_{i}")
            if captured_img: p['img_ans'] = captured_img
        st.divider()

    if st.button("📤 모든 답안 제출", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 채점 결과")
    if not st.session_state.feedback_results:
        with st.spinner("AI 선생님이 채점 중입니다..."):
            all_student_data = []
            for p in st.session_state.problems:
                f_ans = p['ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_ans'] is not None:
                    try:
                        img = Image.open(p['img_ans'])
                        ocr_res = reader.readtext(np.array(img), detail=0)
                        f_ans = " ".join(ocr_res)
                    except: f_ans = "(분석 실패)"
                all_student_data.append(f"Q{p['id']}. 문제: {p['question']} / 학생답: {f_ans}")

            # AI에게 구조화된 응답 요청 (채점 요약 + 틀린 문제 상세 분석)
            analysis_prompt = (
                "너는 친절한 수학 교사야. 아래 15문제를 채점해줘.\n"
                "1. 먼저 각 문제별로 'Q1: O', 'Q2: X' 처럼 정오답 리스트만 짧게 작성해.\n"
                "2. 그 다음, 틀린 문제(X)에 대해서만 '틀린 문제 상세 분석' 섹션을 만들어서 [문제, 분석, 개념 보강, 정답]을 아주 상세하고 길게 설명해줘.\n"
                "3. 맞은 문제(O)에 대해서는 상세 설명을 절대 하지 마.\n\n" + 
                "\n".join(all_student_data)
            )

            full_resp_text = ""
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=MODEL_NAME, contents=analysis_prompt)
                    full_resp_text = resp.text
                    break
                except:
                    if attempt < 2: time.sleep(2); continue
                    else: st.error("AI 통신 지연이 발생했습니다."); st.stop()

            # 정오답 데이터 추출 및 저장
            used_questions = [p['question'] for p in st.session_state.problems]
            used_nums = [re.findall(r'\d+', q)[0] for q in used_questions if re.findall(r'\d+', q)]

            for p in st.session_state.problems:
                # 텍스트에서 O/X 판정 추출
                is_correct = 1 if f"Q{p['id']}: O" in full_resp_text or f"Q{p['id']}. O" in full_resp_text else 0
                st.session_state.feedback_results.append({
                    'id': p['id'], 
                    'is_correct': is_correct, 
                    'raw_feedback': full_resp_text # 전체 응답 저장
                })

                # 오답 시 추천 문제 생성
                if is_correct == 0:
                    try:
                        rec_prompt = f"학생이 '{p['question']}'을 틀렸어. 숫자가 {used_nums}와 완전히 다른 새로운 유사 한글 문제 1개 생성."
                        rec_res = client.models.generate_content(model=MODEL_NAME, contents=rec_prompt).text
                        st.session_state.new_recommendations.append({'ref_id': p['id'], 'q': rec_res, 'ans': "", 'img_ans': None, 'input_type': "⌨️ 타이핑"})
                    except: pass

    # --- UI 표시 영역 ---
    # 1. 요약 리스트 (O/X)
    st.subheader("📊 문항별 정답 요약")
    cols = st.columns(5)
    for i, res in enumerate(st.session_state.feedback_results):
        with cols[i % 5]:
            label = "✅ O" if res['is_correct'] else "❌ X"
            st.markdown(f"**Q{res['id']} : {label}**")
    
    st.divider()

    # 2. 틀린 문제만 상세 피드백 표시
    st.subheader("💡 틀린 문제 상세 클리닉")
    has_wrong = False
    # AI 응답에서 '상세 분석' 부분만 추출하거나 전체를 보여주되 틀린 것만 강조
    st.markdown(st.session_state.feedback_results[0]['raw_feedback']) 
    
    if st.session_state.new_recommendations:
        if st.button("🚀 오답 보완을 위한 추천 문제 풀기", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec['q'])
        rec['input_type'] = st.radio(f"방식 (추천 Q{i+1})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"rectype_{i}", horizontal=True)
        if rec['input_type'] == "⌨️ 타이핑":
            rec['ans'] = st.text_input("정답 입력", value=rec['ans'], key=f"rec_ans_{i}")
        else:
            rec_img = st.camera_input(f"촬영", key=f"reccam_{i}")
            if rec_img: rec['img_ans'] = rec_img
        st.divider()
    if st.button("🏁 최종 진단 보고서", type="primary"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 최종 성취도 분석")
    if 'bkt_report' not in st.session_state:
        status, acc, gain = diagnose_learning_status(st.session_state.feedback_results)
        try:
            report = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정확도:{acc*100:.1f}%. 한글로 학습 처방 조언해줘.").text
            st.session_state.bkt_report = report
        except: st.session_state.bkt_report = "리포트 생성 실패"
        st.session_state.bkt_status = status
    
    st.success(f"### 최종 성취 등급: {st.session_state.bkt_status}")
    st.markdown(st.session_state.bkt_report)
    if st.button("🔄 처음으로 돌아가기"):
        st.session_state.clear(); st.rerun()
