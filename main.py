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
    """OCR 엔진을 캐싱하여 리소스 낭비와 inotify 에러를 방지합니다."""
    return easyocr.Reader(['ko', 'en'], gpu=False)

@st.cache_resource
def load_rf_model(path):
    """랜덤 포레스트 모델을 캐싱합니다."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

reader = load_ocr_reader()

# API 설정 (사용자 지정: Gemini 2.5 Flash)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash" 

# 경로 설정 (사용자 디렉토리 및 파일명 준수)
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
    """영어 문제를 번역하며 f-string 백슬래시 문법 오류를 해결한 버전"""
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
        # ✅ SyntaxError 해결: f-string 내부에서 직접 re.findall(r'\d+')를 쓰지 않음
        fallback_list = []
        for t in en_list:
            nums = re.findall(r'\d+', t)
            target_num = nums[0] if nums else "수"
            fallback_list.append(f"{target_num}를 소인수분해하시오.")
        return fallback_list

def diagnose_status(results):
    """BKT 지표와 랜덤포레스트 기반의 고도화된 진단 로직"""
    corrects = [r.get('is_correct', 0) for r in results]
    actual_acc = (sum(corrects) / len(results)) * 100 if results else 0.0
    changes = sum(1 for i in range(len(corrects)-1) if corrects[i] != corrects[i+1])
    stability = 1 - (changes / (len(corrects)-1)) if len(corrects) > 1 else 1.0

    model = load_rf_model(RF_MODEL_PATH)
    if not model: return "모델 파일 없음", actual_acc, stability
        
    try:
        init_k = np.mean(corrects[:5]) * 0.5 
        final_k = np.mean(corrects[-5:]) * 0.9 
        gain = max(0, final_k - init_k)
        var = np.var(corrects)
        
        df_input = pd.DataFrame([[init_k, final_k, gain, actual_acc/100, var]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        status = model.predict(df_input)[0]

        # ✅ 사용자 요청 등급 보정 기준
        if actual_acc == 100.0:
            status = "Master (완벽)"
        elif actual_acc <= 30.0 or (actual_acc < 60.0 and stability <= 0.3):
            # 거의 못 맞히거나 정오답 패턴이 지나치게 불안정한 경우
            status = "Unstable (불안정)"
            
        return status, actual_acc, stability
    except:
        return "진단 처리 오류", actual_acc, stability

# ===============================
# 3. 단계별 UI 흐름
# ===============================

# [Step 0] 시작
if st.session_state.step == 0:
    st.title("🎓 취약점 캐쳐: AI 학습 진단")
    st.write("소인수분해를 통해 나의 학습 상태를 정밀하게 진단받으세요.")
    if st.button("🚀 오늘 학습 시작", type="primary", use_container_width=True):
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            pool = df['generated_problem_english'].dropna().unique().tolist()
            # 중복 숫자 문제 제거
            unique_numbered_dict = {}
            for t in pool:
                nums = re.findall(r'\d+', t)
                if nums: unique_numbered_dict[nums[0]] = t
            clean_pool = list(unique_numbered_dict.values())

            if len(clean_pool) >= 15:
                selected_en = random.sample(clean_pool, 15)
                translated_ko = translate_problems(selected_en)
                st.session_state.problems = [
                    {'id': i+1, 'question': q, 'ans': "", 'img_bytes': None, 'input_type': "⌨️ 타이핑"} 
                    for i, q in enumerate(translated_ko)
                ]
                st.session_state.step = 1
                st.rerun()

# [Step 1] 문제 풀이 (사진 데이터 영구 보존)
elif st.session_state.step == 1:
    st.title("📝 1차 학습")
    for i, p in enumerate(st.session_state.problems):
        st.subheader(f"Q{p['id']}. {p['question']}")
        p['input_type'] = st.radio(f"방식_{i}", ["⌨️ 타이핑", "📸 사진 찍기"], key=f"t_{i}", horizontal=True, label_visibility="collapsed")
        
        if p['input_type'] == "⌨️ 타이핑":
            p['ans'] = st.text_input(f"답안_{i}", value=p['ans'], key=f"a_{i}", label_visibility="collapsed")
        else:
            cam_file = st.camera_input(f"촬영_{i}", key=f"cam_{i}", label_visibility="collapsed")
            if cam_file:
                # ✅ 사진을 바이트로 즉시 저장하여 단계 이동 시 증발 방지
                st.session_state.problems[i]['img_bytes'] = cam_file.getvalue()
                st.success(f"Q{p['id']} 사진 기록 완료")
        st.divider()

    if st.button("📤 답안 일괄 제출", type="primary", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

# [Step 2] 채점 및 틀린 문제만 피드백
elif st.session_state.step == 2:
    st.title("🔍 분석 및 피드백")
    if not st.session_state.feedback_results:
        with st.spinner("AI가 틀린 문제만 골라 분석하고 있습니다..."):
            student_summary = []
            for p in st.session_state.problems:
                f_ans = p['ans']
                if p['input_type'] == "📸 사진 찍기" and p['img_bytes']:
                    try:
                        img = Image.open(io.BytesIO(p['img_bytes']))
                        ocr_res = reader.readtext(np.array(img), detail=0)
                        f_ans = " ".join(ocr_res) if ocr_res else "(글자 인식 불가)"
                    except: f_ans = "(이미지 오류)"
                student_summary.append(f"Q{p['id']}: {f_ans} (문제: {p['question']})")

            # ✅ 강력한 오답 전용 피드백 프롬프트
            prompt = (
                "수학 교사로서 채점해줘.\n"
                "1. [채점]: 'Q번호: O/X' 리스트를 반드시 작성해.\n"
                "2. [분석]: 반드시 'X'로 표시된 틀린 문제에 대해서만 '사고의 사슬' 기법을 사용해 원인을 분석해.\n"
                "⚠️ 맞은 문제(O)는 절대 피드백이나 설명을 제공하지 마."
            )
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt + "\n\n" + "\n".join(student_summary))
            st.session_state.full_feedback = resp.text

            # 추천 문제 생성
            for p in st.session_state.problems:
                match = re.search(f"Q{p['id']}:?\s*([OX0])", resp.text, re.IGNORECASE)
                is_ok = 1 if match and match.group(1).upper() in ['O', '0'] else 0
                st.session_state.feedback_results.append({'id': p['id'], 'is_correct': is_ok})
                if is_ok == 0:
                    rec_p = client.models.generate_content(model=MODEL_NAME, contents=f"학생이 '{p['question']}'을 틀렸어. 숫자만 바꾼 유사 문제 1개를 생성해.").text
                    st.session_state.new_recommendations.append({'q': rec_p, 'ans': "", 'img_bytes': None})

    # UI 출력
    cols = st.columns(5)
    for i, res in enumerate(st.session_state.feedback_results):
        with cols[i % 5]:
            st.write(f"Q{res['id']} {'✅' if res['is_correct'] else '❌'}")
    
    st.divider()
    st.markdown(st.session_state.full_feedback)
    
    if st.session_state.new_recommendations:
        if st.button(f"🚀 추천 문제 ({len(st.session_state.new_recommendations)}개) 풀기", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# [Step 3] 추천 문제
elif st.session_state.step == 3:
    st.title("🎯 맞춤 보완 학습")
    for i, rec in enumerate(st.session_state.new_recommendations):
        st.info(f"💡 {rec['q']}")
        rec['ans'] = st.text_input(f"답안 입력 {i}", key=f"ra_{i}")
        st.divider()
    
    if st.button("🏁 최종 성취도 리포트 확인", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

# [Step 4] 성취도 리포트 (Master/Unstable 반영)
elif st.session_state.step == 4:
    st.title("🏆 종합 성취도 리포트")
    status, acc, stab = diagnose_status(st.session_state.feedback_results)
    
    st.success(f"### 성취 등급: {status}")
    st.metric("최종 정답률", f"{acc:.1f}%")
    st.info(f"안정성 지표: {stab*100:.1f}%")
    
    try:
        rep = client.models.generate_content(model=MODEL_NAME, contents=f"상태:{status}, 정답률:{acc}%. 학생에게 줄 조언을 한글로.").text
        st.write(rep)
    except: st.write("리포트를 불러올 수 없습니다.")
    
    if st.button("🔄 처음으로 돌아가기"):
        st.session_state.clear(); st.rerun()
