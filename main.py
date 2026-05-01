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
    st.session_state.run_id = str(int(time.time() * 1000)) # 밀리초 단위 고유 ID

# ===============================
# 2. 강력한 번역 및 진단 로직
# ===============================

def translate_problems_batch(en_list):
    """15문제를 한 번의 API 호출로 모두 번역 (에러 방지 핵심)"""
    combined_query = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(en_list)])
    prompt = (
        "너는 수학 교사야. 아래의 영문 문제들을 순서대로 한국어 '-하시오' 체로 번역해줘.\n"
        "영어는 절대 섞지 말고 한국어 문장만 번역해서 15개를 나열해줘.\n\n"
        f"{combined_query}"
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        translated_lines = resp.text.strip().split('\n')
        # 번역 결과가 리스트와 맞지 않을 경우 비상 로직 가동
        results = []
        for i, en_text in enumerate(en_list):
            try:
                line = [l for l in translated_lines if str(i+1) in l[:4]][0]
                results.append(re.sub(r'^\d+\.\s*', '', line))
            except:
                # 비상: 숫자만 추출해서 문장 강제 생성
                num = re.search(r'\d+', en_text).group() if re.search(r'\d+', en_text) else "제시된 수"
                results.append(f"{num}을 소인수분해하시오.")
        return results
    except Exception as e:
        # 전체 실패 시 숫자 기반 강제 생성
        return [f"{re.search(r'\d+', t).group() if re.search(r'\d+', t) else '수'}를 소인수분해하시오." for t in en_list]

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
                
                with st.spinner("AI가 15문제를 한 번에 번역하고 있습니다..."):
                    translated_ko = translate_problems_batch(selected_en)
                    st.session_state.problems = [{'id': i+1, 'question': q, 'ans': ""} for i, q in enumerate(translated_ko)]
                
                st.session_state.run_id = str(int(time.time() * 1000))
                st.session_state.step = 1
                st.rerun()
        else:
            st.error("데이터셋 파일(csv)을 찾을 수 없습니다.")

elif st.session_state.step == 1:
    st.title("📝 1차 학습: 15문제")
    st.warning("⚠️ 자동 완성 기록을 방지하기 위해 입력창 ID가 매번 변경됩니다.")
    
    for i, p in enumerate(st.session_state.problems):
        st.markdown(f"### **Q{i+1}.**")
        st.info(p.get('question'))
        
        # 💡 자동완성 네모 칸 제거 핵심: 라벨에 타임스탬프와 랜덤값을 섞음
        # 브라우저는 라벨이 1글자만 달라도 완전히 새로운 입력창으로 인식함
        rand_val = random.randint(100, 999)
        dynamic_label = f"정답 입력 (세션:{st.session_state.run_id[-4:]}-번호:{i+1}-난수:{rand_val})"
        
        p['ans'] = st.text_input(dynamic_label, key=f"ans_{i}_{st.session_state.run_id}_{rand_val}")
        st.divider()

    if st.button("📤 답안 제출 및 상세 분석", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.title("🔍 채점 결과 및 AI 피드백")
    if not st.session_state.feedback_results:
        with st.spinner("AI 선생님이 전체 답안을 정밀 분석 중입니다..."):
            for p in st.session_state.problems:
                prompt = f"문제: {p['question']}\n학생 답: {p['ans']}\n판정(O/X), 논리분석, 개념보강, 정답공개 순으로 작성해줘."
                resp = client.models.generate_content(model=MODEL_NAME, contents=prompt).text
                is_correct = 1 if resp.strip().startswith('O') else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_correct, 'content': resp})
                
                if not is_correct:
                    rec_q = client.models.generate_content(model=MODEL_NAME, contents=f"'{p['question']}'의 원리를 다루는 한글 문제를 하나 만들어줘.").text
                    st.session_state.new_recommendations.append({'ref_id': p['id'], 'q': rec_q, 'ans': ""})

    for res in st.session_state.feedback_results:
        if res['is_correct']: st.success(f"✅ Q{res['id']}\n{res['content']}")
        else: st.error(f"❌ Q{res['id']}\n{res['content']}")

    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제로 취약점 보완하기", type="primary"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.title("🎯 맞춤 추천 문제")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec.get('q'))
        # 추천 문제도 자동완성 방지 적용
        r_val = random.randint(10, 99)
        rec['ans'] = st.text_input(f"답안 작성 (코드:{st.session_state.run_id[-2:]}-{r_val})", key=f"rec_{i}_{st.session_state.run_id}_{r_val}")

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
