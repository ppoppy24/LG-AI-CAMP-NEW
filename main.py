import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import joblib
import os
import time
from google import genai

# 1. 환경 설정
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

# API 설정 (Gemini 3 Flash 환경)
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
                    st.session_state.problems = [{'id': i+1, 'question': q, 'ans': ""} for i, q in enumerate(translated_ko)]
                st.session_state.run_id = str(int(time.time() * 1000))
                st.session_state.step = 1
                st.rerun()

elif st.session_state.step == 1:
    st.title("📝 1차 학습")
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p.get('question'))
        
        # 🛡️ 자동 완성 방지: "미끼" 입력창과 "진짜" 입력창의 분리
        # 가짜 입력창을 화면 밖으로 숨겨 브라우저의 자동완성 데이터를 가로챕니다.
        st.markdown('<input type="text" style="display:none;" name="fake_input_remember_me">', unsafe_allow_html=True)
        
        st.write("✍️ 정답을 입력하세요:")
        # 라벨을 완전히 무작위화하고 숨깁니다.
        r_str = "".join(random.choices("lmnop", k=8))
        p['ans'] = st.text_input(
            label=f"L_{r_str}_{i}", 
            key=f"K_{st.session_state.run_id}_{i}_{r_str}",
            label_visibility="collapsed" # 라벨 숨김
        )
        st.divider()

    if st.button("📤 답안 제출", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 분석 결과")
    if not st.session_state.feedback_results:
        with st.spinner("AI가 채점 중..."):
            for p in st.session_state.problems:
                prompt = f"문제: {p['question']}\n학생 답: {p['ans']}\nO/X 판정, 분석, 개념보강, 정답공개 순으로 작성."
                resp = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
                is_correct = 1 if resp.strip().startswith('O') else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'content': resp})
                if not is_correct:
                    rec_q = client.models.generate_content(model=MODEL_NAME, contents=f"'{p['question']}' 유사 한글 문제 생성.").text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'q': rec_q, 'ans': ""})

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
        st.write("✍️ 답안:")
        # 추천 문제도 동일한 방어막 적용
        st.markdown('<input type="text" style="display:none;" name="fake_rec_input">', unsafe_allow_html=True)
        r_rec = "".join(random.choices("qrs", k=5))
        rec['ans'] = st.text_input(label=f"R_{r_rec}", key=f"RE_{i}_{r_rec}", label_visibility="collapsed")

    if st.button("🏁 최종 진단"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 최종 성취도")
    status, acc, gain = diagnose_learning_status(st.session_state.feedback_results)
    report = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정확도:{acc*100:.1f}%. 조언 한글로.").text
    st.success(f"### 성취 등급: {status}")
    st.markdown(report)
    if st.button("🔄 처음으로"):
        st.session_state.clear()
        st.rerun()
