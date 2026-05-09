import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import joblib
import os
import easyocr
from PIL import Image
import io
from google import genai

# 1. 환경 설정 및 에러 방지 캐싱
st.set_page_config(page_title="AI BKT 학습 시스템", layout="centered")

@st.cache_resource
def load_ocr_reader():
    """EasyOCR 리더를 한 번만 로드하여 inotify 인스턴스 낭비를 방지합니다."""
    return easyocr.Reader(['ko', 'en'], gpu=False)

@st.cache_resource
def load_rf_model(path):
    """랜덤 포레스트 모델을 캐싱합니다."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

reader = load_ocr_reader()

# API 설정 (Gemini 2.5 Flash 고정)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash" 

# 경로 설정
BASE_DIR = "/mount/src/lg-ai-camp-new"
RF_MODEL_PATH = os.path.join(BASE_DIR, 'bkt_rf__model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'bkt_training_dataset_english_problem.csv')

# 세션 상태 초기화
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.problems = []
    st.session_state.feedback_results = []
    st.session_state.new_recommendations = []
    st.session_state.full_feedback = ""

# ===============================
# 2. 핵심 로직 함수
# ===============================

def translate_problems(en_list):
    prompt = (
        "수학 선생님으로서 다음 영어 문제들을 한국어로 번역해줘.\n"
        "'-하시오' 체를 사용하고, 인사말 없이 번호와 문장만 나열해.\n\n" + 
        "\n".join([f"{i+1}. {t}" for i, t in enumerate(en_list)])
    )
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        lines = resp.text.strip().split('\n')
        return [re.sub(r'^\d+\.\s*', '', l).strip() for l in lines if l and "번역" not in l[:10]]
    except:
        return [f"{re.findall(r'\d+', t)[0] if re.findall(r'\d+', t) else '수'}를 소인수분해하시오." for t in en_list]

def diagnose_status(results):
    corrects = [r.get('is_correct', 0) for r in results]
    actual_acc = (sum(corrects) / len(results)) * 100 if results else 0.0
    changes = sum(1 for i in range(len(corrects)-1) if corrects[i] != corrects[i+1])
    stability = 1 - (changes / (len(corrects)-1)) if len(corrects) > 1 else 1.0

    model = load_rf_model(RF_MODEL_PATH)
    if not model: return "모델 없음", actual_acc, stability
        
    try:
        init_k = np.mean(corrects[:5]) * 0.5 
        final_k = np.mean(corrects[-5:]) * 0.9 
        gain = max(0, final_k - init_k)
        var = np.var(corrects)
        
        df_input = pd.DataFrame([[init_k, final_k, gain, actual_acc/100, var]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        status = model.predict(df_input)[0]

        # 등급 보정 가드레일
        if actual_acc == 100.0: status = "Master (완벽)"
        elif actual_acc <= 30.0 or (actual_acc < 60.0 and stability <= 0.3):
            status = "Unstable (불안정)"
        return status, actual_acc, stability
    except:
        return "진단 오류", actual_acc, stability

# ===============================
# 3. 단계별 UI
# ===============================

# [Step 0] 시작 화면
if st.session_state.step == 0:
    st.title("🎓 취약점 캐쳐: AI 학습 진단")
    st.write("소인수분해 문제를 풀고 나만의 학습 등급을 확인하세요.")
    if st.button("🚀 학습 시작하기", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            unique_numbered_dict = {re.findall(r'\d+', t)[0]: t for t in pool if re.findall(r'\d+', t)}
            clean_pool = list(unique_numbered_dict.values())
            if len(clean_pool) >= 15:
                selected_en = random.sample(clean_pool, 15)
                st.session_state.problems = [{'id': i+1, 'question': q, 'ans': "", 'img_bytes': None, 'input_type': "⌨️ 타이핑"} for i, q in enumerate(translate_problems(selected_en))]
                st.session_state.step = 1
                st.rerun()

# [Step 1] 문제 풀이 (사진 증발 방지 로직 포함)
elif st.session_state.step == 1:
    st.title("📝 소인수분해 연습")
    for i, p in enumerate(st.session_state.problems):
        st.subheader(f"Q{p['id']}. {p['question']}")
        p['input_type'] = st.radio(f"입력 방식 {i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"t_{i}", horizontal=True)
        
        if p['input_type'] == "⌨️ 타이핑":
            p['ans'] = st.text_input(f"정답 입력 {i}", value=p['ans'], key=f"a_{i}")
        else:
            cam_file = st.camera_input(f"풀이 촬영 {i}", key=f"cam_{i}")
            if cam_file:
                st.session_state.problems[i]['img_bytes'] = cam_file.getvalue()
                st.success("사진이 저장되었습니다.")
        st.divider()

    if st.button("📤 답안 제출 및 분석", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

# [Step 2] 채점 및 오답 전용 피드백
elif st.session_state.step == 2:
    st.title("🔍 채점 결과 및 분석")
    if not st.session_state.feedback_results:
        with st.spinner("AI가 틀린 문제만 쏙쏙 골라 분석 중입니다..."):
            student_summary = []
            for p in st.session_state.problems:
                final_ans = p['ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_bytes']:
                    try:
                        img = Image.open(io.BytesIO(p['img_bytes']))
                        ocr_res = reader.readtext(np.array(img), detail=0)
                        final_ans = " ".join(ocr_res) if ocr_res else "(인식 실패)"
                    except: final_ans = "(이미지 오류)"
                student_summary.append(f"Q{p['id']}: {final_ans} (문제: {p['question']})")

            # 오답 전용 피드백 프롬프트 (사고의 사슬 적용)
            prompt = (
                "너는 친절한 수학 교사야. 다음 학생의 답안을 채점해줘.\n"
                "1. [채점 리스트]: 'Q1: O', 'Q2: X' 형태로 모든 문항의 정오답을 표시해.\n"
                "2. [오답 피드백]: 'X'로 표시된 틀린 문제에 대해서만 '사고의 사슬' 기법을 써서 왜 틀렸는지 분석해줘.\n"
                "⚠️ 주의: 맞은 문제(O)는 절대 언급하거나 설명하지 마."
            )
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt + "\n\n" + "\n".join(student_summary))
            st.session_state.full_feedback = resp.text

            # 결과 파싱 및 추천 문제 생성
            for p in st.session_state.problems:
                match = re.search(f"Q{p['id']}:?\s*([OX0])", resp.text, re.IGNORECASE)
                is_ok = 1 if match and match.group(1).upper() in ['O', '0'] else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_ok})
                if is_ok == 0:
                    rec_p = client.models.generate_content(model=MODEL_NAME, contents=f"'{p['question']}'을 틀린 학생을 위한 숫자만 바꾼 유사 문제 1개를 '문제: '로 시작해서 만들어줘.").text
                    st.session_state.new_recommendations.append({'q': rec_p, 'ans': "", 'img_bytes': None})

    # 결과 UI
    cols = st.columns(5)
    for i, res in enumerate(st.session_state.feedback_results):
        with cols[i % 5]:
            st.write(f"Q{res['id']} {'✅' if res['is_correct'] else '❌'}")
    
    st.divider()
    st.markdown(st.session_state.full_feedback)
    
    if st.session_state.new_recommendations:
        if st.button("🚀 추천 문제 풀러 가기", type="primary"):
            st.session_state.step = 3
            st.rerun()

# [Step 3] 추천 문제
elif st.session_state.step == 3:
    st.title("🎯 맞춤 보완 학습")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(rec['q'])
        rec['ans'] = st.text_input(f"정답 입력 {i}", key=f"ra_{i}")
        st.divider()
    
    if st.button("🏁 최종 진단 결과 보기", type="primary"):
        st.session_state.step = 4
        st.rerun()

# [Step 4] 최종 리포트 (Master/Unstable 반영)
elif st.session_state.step == 4:
    st.title("🏆 최종 학습 리포트")
    status, acc, stab = diagnose_status(st.session_state.feedback_results)
    
    st.metric("학습 상태", status)
    st.progress(acc / 100, text=f"정답률: {acc:.1f}%")
    st.write(f"안정성 지표: {stab*100:.1f}% (패턴이 일관될수록 높습니다)")
    
    try:
        report = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정답률:{acc}%인 학생에게 주는 격려와 조언을 한국어로 작성해줘.").text
        st.write(report)
    except: st.write("진단서를 불러오는 중입니다...")
    
    if st.button("🔄 처음으로 돌아가기"):
        st.session_state.clear(); st.rerun()
