import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import cv2
import easyocr
import joblib
import os
import time
from PIL import Image
from google import genai

# 1. 환경 설정 및 초기화
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

# OCR 모델 로드 (캐싱)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr()

# API 설정 (Streamlit Cloud의 Settings > Secrets에 GEMINI_API_KEY 입력 필요)
API_KEY = st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# GitHub 루트 경로에 있는 파일 이름
RF_MODEL_PATH = 'bkt_rf_model.pkl' 
DATA_PATH = 'bkt_training_dataset_english_problem.csv'

# ===============================
# 2. RF 모델 진단 로직
# ===============================
def diagnose_learning_status(results):
    if not os.path.exists(RF_MODEL_PATH):
        return "MODEL_FILE_NOT_FOUND", 0, 0, 0

    try:
        model = joblib.load(RF_MODEL_PATH)
        correctness = [r['is_correct'] for r in results]
        accuracy = sum(correctness) / len(results) if results else 0
        initial_k = results[0]['is_correct'] * 0.4 if results else 0
        final_k = results[-1]['is_correct'] * 0.8 if results else 0
        learning_gain = max(0, final_k - initial_k)
        variance = np.var(correctness) if len(correctness) > 1 else 0.0

        input_df = pd.DataFrame([[initial_k, final_k, learning_gain, accuracy, variance]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        
        status = model.predict(input_df)[0]
        return status, accuracy, learning_gain, initial_k
    except Exception as e:
        return f"ERROR_{str(e)}", 0, 0, 0

# ===============================
# 3. 수식 및 문제 유틸리티
# ===============================
def ocr_read(image_file):
    img = Image.open(image_file)
    img_np = np.array(img)
    result = reader.readtext(img_np)
    return " ".join([r[1] for r in result])

def extract_number(text):
    nums = re.findall(r"\d+", str(text))
    return int(nums[-1]) if nums else None

def prime_factorization(n):
    factors = []; d = 2
    while d*d <= n:
        while n % d == 0: factors.append(d); n //= d
        d += 1
    if n > 1: factors.append(n)
    return factors

def normalize(text):
    text = str(text).lower().replace("*","x").replace("×","x").replace(" ","")
    try: return sorted([int(x) for x in text.split("x") if x.isdigit()])
    except: return []

def generate_recommend_num(original_num):
    factors = prime_factorization(original_num); primes = [2, 3, 5, 7]
    if len(factors) > 2 and random.choice([True, False]): factors.pop(random.randint(0, len(factors)-1))
    else: factors.append(random.choice(primes))
    new_num = 1
    for f in factors: new_num *= f
    return new_num if new_num > 1 else original_num + 2

# ===============================
# 4. 세션 관리 및 흐름 제어
# ===============================
if 'step' not in st.session_state:
    st.session_state.step = 0 # 0: 시작, 1: 문제풀이, 2: 피드백/추천, 3: 최종진단
    st.session_state.problems = []
    st.session_state.results = []
    st.session_state.recommend_results = []
    st.session_state.run_id = str(time.time())

# ===============================
# 5. 메인 UI (6단계 로직 반영)
# ===============================
st.title("🎓 AI BKT 맞춤형 학습 시스템")

# [단계 0] 시작 화면
if st.session_state.step == 0:
    st.write("LG-AI-CAMP-NEW 데이터셋에서 문제를 불러옵니다.")
    if st.button("🚀 학습 시작하기 (15문제 생성)", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            # 중복 제거 후 15문제 샘플링
            sample = df.drop_duplicates(subset=['generated_problem_english']).sample(n=min(15, len(df))).reset_index(drop=True)
            st.session_state.problems = sample.to_dict('records')
            st.session_state.step = 1
            st.rerun()
        else:
            st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

# [단계 1] 기초 문제 풀이
elif st.session_state.step == 1:
    st.header("📝 1차 학습: 오늘의 15문제")
    
    for i, row in enumerate(st.session_state.problems):
        with st.container():
            num = extract_number(row["generated_problem_english"])
            st.markdown(f"### **Q{i+1}.**")
            st.info(f"{num}을 소인수분해하시오.")
            
            # [이미지 UI 반영] 답변 방식 선택
            mode = st.radio(f"답변 방식 (Q{i+1})", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"mode_{i}", horizontal=True)
            
            # ✅ 수정 후
            if mode == "⌨️ 타이핑":
                st.text_input("정답 입력", key=f"ans_{i}_{st.session_state.run_id}", placeholder="예: 2x2x3")
            else:
                img_file = st.camera_input(f"Capture (Q{i+1})", key=f"cam_{i}")
                if img_file:
                    ocr_text = ocr_read(img_file)
                    st.text_input("인식 결과 (수정 가능)", value=ocr_text, key=f"ocr_{i}")
            st.divider()

    if st.button("📤 답안 제출 및 채점", type="primary", use_container_width=True):
        results = []
        for i, row in enumerate(st.session_state.problems):
            num = extract_number(row["generated_problem_english"])
            u_ans = st.session_state.get(f"ans_{i}") if st.session_state[f"mode_{i}"] == "⌨️ 타이핑" else st.session_state.get(f"ocr_{i}")
            
            correct_list = prime_factorization(num)
            is_correct = 1 if normalize(u_ans) == sorted(correct_list) else 0
            results.append({"num": num, "is_correct": is_correct, "user_ans": u_ans, "correct_ans": 'x'.join(map(str, correct_list))})
        
        st.session_state.results = results
        st.session_state.step = 2
        st.rerun()

# [단계 2] 채점 피드백 및 추천 문제
elif st.session_state.step == 2:
    st.header("🔍 채점 결과 및 상세 분석")
    wrong_list = [r for r in st.session_state.results if r["is_correct"] == 0]
    
    # 결과 출력
    for res in st.session_state.results:
        if res['is_correct']: st.success(f"✅ Q. {res['num']} : 정답입니다!")
        else: st.error(f"❌ Q. {res['num']} : 오답 (입력: {res['user_ans']} / 정답: {res['correct_ans']})")

    if not wrong_list:
        st.balloons()
        st.success("🎉 모든 문제를 맞혔습니다!")
        if st.button("처음으로"): st.session_state.clear(); st.rerun()
    else:
        # AI 피드백 생성
        summary = "\n".join([f"문제: {r['num']}, 학생답: {r['user_ans']}, 정답: {r['correct_ans']}" for r in wrong_list])
        with st.spinner("AI 선생님의 오답 분석 중..."):
# ✅ 수정 후 (model= 키워드 추가)
            resp = client.models.generate_content(
            model=MODEL_NAME, 
            contents=f"수학 교사로서 다음 오답을 분석하고 격려해줘: {summary}"
)
        
        st.divider()
        st.subheader("💡 오답 맞춤 추천 문제")
        for j, wr in enumerate(wrong_list):
            new_num = generate_recommend_num(wr["num"])
            st.write(f"**추천 Q{j+1}.** {new_num}을 소인수분해하시오.")
            # ✅ 수정 후
            st.text_input(f"답안 입력", key=f"rec_ans_{j}_{st.session_state.run_id}")

        if st.button("🏁 최종 제출 및 BKT 진단", type="primary"):
            rec_results = []
            for j, wr in enumerate(wrong_list):
                new_num = generate_recommend_num(wr["num"])
                u_ans = st.session_state.get(f"rec_ans_{j}")
                correct_list = prime_factorization(new_num)
                is_correct = 1 if normalize(u_ans) == sorted(correct_list) else 0
                rec_results.append({"num": new_num, "is_correct": is_correct, "user_ans": u_ans})
            
            st.session_state.recommend_results = rec_results
            st.session_state.step = 3
            st.rerun()

# [단계 3] 최종 BKT 진단
elif st.session_state.step == 3:
    st.header("🏆 최종 학습 진단 보고서")
    
    status, acc, gain, init = diagnose_learning_status(st.session_state.results + st.session_state.recommend_results)
    
    with st.spinner("BKT 알고리즘으로 성취도를 분석 중입니다..."):
        prompt = f"상태:{status}, 정확도:{acc*100:.1f}%, 성장도:{gain:.2f}. 이 수치를 바탕으로 학생에게 A~E 등급을 부여하고 조언해줘."
        resp = client.models.generate_content(MODEL_NAME, contents=prompt)
        st.markdown(resp.text)
        st.balloons()

    if st.button("🔄 처음으로"):
        st.session_state.clear()
        st.rerun()
