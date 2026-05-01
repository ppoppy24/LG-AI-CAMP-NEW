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

# API 및 모델 설정
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
    st.session_state.run_id = str(int(time.time() * 100))

# ===============================
# 2. 강력한 번역 및 진단 로직 (SyntaxError 해결)
# ===============================

def translate_problems_batch(en_list):
    """15문제를 한 번에 번역 (에러 방지 및 속도 향상)"""
    combined_query = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(en_list)])
    prompt = (
        "너는 수학 선생님이야. 아래 영문 문제들을 순서대로 한국어 '-하시오' 체로 번역해줘.\n"
        "영어는 절대 섞지 말고 한국어 문장만 번역해서 15개를 나열해줘.\n\n"
        f"{combined_query}"
    )
    
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        translated_lines = resp.text.strip().split('\n')
        
        results = []
        for i, en_text in enumerate(en_list):
            try:
                # 해당 번호로 시작하는 라인 찾기
                line = [l for l in translated_lines if str(i+1) in l[:5]][0]
                results.append(re.sub(r'^\d+[\.\s]*', '', line).strip())
            except:
                # 비상: 숫자 추출 로직 (f-string 외부에서 처리하여 SyntaxError 방지)
                nums = re.findall(r'\d+', en_text)
                n = nums[0] if nums else "수"
                results.append(n + "를 소인수분해하시오.")
        return results
    except:
        # 전체 API 실패 시 비상 처리
        fallback = []
        for t in en_list:
            nums = re.findall(r'\d+', t)
            n = nums[0] if nums else "수"
            fallback.append(n + "를 소인수분해하시오.")
        return fallback

def diagnose_learning_status(results):
    """RF 모델 진단"""
    if not os.path.exists(RF_MODEL_PATH): return "MODEL_MISSING", 0, 0
    try:
        model = joblib.load(RF_MODEL_PATH)
        corrects = [r.get('is_correct', 0) for r in results]
        acc = sum(corrects) / len(results) if results else 0
        init_k, final_k = corrects[0]*0.4, corrects[-1]*0.8
        var = np.var(corrects) if len(corrects) > 1 else 0.0
        input_df = pd.DataFrame([[init_k, final_k, max(0, final_k-init_k), acc, var]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        return model.predict(input_df)[0], acc, max(0, final_k-init_k)
    except: return "ERROR", 0, 0

# ===============================
# 3. 단계별 UI
# ===============================

if st.session_state.step == 0:
    st.title("🎓 AI 맞춤형 BKT 학습 시스템")
    if st.button("🚀 오늘의 문제 시작하기 (15문항)", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            if len(pool) >= 15:
                selected_en = random.sample(pool, 15)
                with st.spinner("AI가 15문제를 번역 중입니다..."):
                    translated_ko = translate_problems_batch(selected_en)
                    st.session_state.problems = [{'id': i+1, 'question': q, 'ans': ""} for i, q in enumerate(translated_ko)]
                st.session_state.run_id = str(int(time.time() * 100))
                st.session_state.step = 1
                st.rerun()
        else:
            st.error("데이터셋(csv) 파일을 찾을 수 없습니다.")

elif st.session_state.step == 1:
    st.title("📝 1차 학습: 15문제")
    
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p.get('question'))
        
        # 💡 자동완성(검은 네모) 방지 핵심: 
        # 1. 라벨에 무작위 유니코드(자음/모음 등)를 섞어 브라우저가 인식 못하게 함
        # 2. key를 매번 완전히 다르게 생성
        random_char = random.choice(["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ"])
        dynamic_label = f"정답 입력 {random_char} (코드:{st.session_state.run_id[-4:]}-{i})"
        
        p['ans'] = st.text_input(
            dynamic_label, 
            key=f"input_{i}_{st.session_state.run_id}_{random.randint(100,999)}",
            placeholder="예: 2x3x5"
        )
        st.divider()

    if st.button("📤 모든 답안 제출 및 분석", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 채점 결과 및 AI 피드백")
    if not st.session_state.feedback_results:
        with st.spinner("AI 선생님이 답안을 정밀 분석 중입니다..."):
            for p in st.session_state.problems:
                prompt = f"문제: {p['question']}\n학생 답: {p['ans']}\n판정(O/X), 분석, 개념보강, 정답공개 순으로 작성해줘."
                resp = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
                is_correct = 1 if resp.strip().startswith('O') else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'content': resp})
                
                if not is_correct:
                    rec_q = client.models.generate_content(model=MODEL_NAME, contents=f"'{p['question']}'과 비슷한 한글 문제를 하나 만들어줘.").text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'q': rec_q, 'ans': ""})

    for res in st.session_state.feedback_results:
        if res['is_correct']: st.success(f"✅ Q{res['id']}\n{res['content']}")
        else: st.error(f"❌ Q{res['id']}\n{res['content']}")

    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제 풀기", type="primary"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec.get('q'))
        # 추천 문제도 자동완성 차단
        r_char = random.choice(["⭐", "🔹", "🔸", "📍"])
        rec['ans'] = st.text_input(
            f"추천 답안 {r_char} ({st.session_state.run_id[-2:]}-{i})", 
            key=f"rec_{i}_{st.session_state.run_id}_{random.randint(10,99)}"
        )

    if st.button("🏁 최종 진단 보고서", type="primary"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.title("🏆 최종 성취도 진단")
    status, acc, gain = diagnose_learning_status(st.session_state.feedback_results)
    report = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정확도:{acc*100:.1f}%. 등급과 조언을 한글로 해줘.").text
    
    st.success(f"### 분석 결과: {status}")
    st.markdown(report)
    if st.button("🔄 처음으로"):
        st.session_state.clear()
        st.rerun()
