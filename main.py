import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import nest_asyncio
import random
from PIL import Image

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 통합 문제 학습기", page_icon="📸", layout="centered")

# 2. 보안 설정 및 모델 로드 (Gemini 2.5 Flash)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # 사용자의 요청에 따라 모델명을 gemini-2.5-flash로 설정
    # (환경에 따라 404 에러 발생 시 gemini-2.0-flash로 변경하여 사용하세요)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ API 키를 설정해주세요. (Streamlit Cloud의 Secrets 메뉴)")
    st.stop()

# 3. 데이터 로드 (문제 은행)
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    return pd.read_csv(data_path) if os.path.exists(data_path) else None

df = load_data()

# 4. 세션 상태 관리 (통합 문제 리스트)
if 'master_problems' not in st.session_state:
    st.session_state.master_problems = []

# 5. 메인 화면 구성
st.title("🎓 AI 통합 학습 시스템")
st.write("사진을 찍거나 데이터셋에서 문제를 가져와 한글로 학습하세요.")
st.divider()

# --- 상단: 문제 추가 섹션 (OCR & 자동 배정) ---
st.subheader("➕ 문제 추가하기")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### **📸 사진으로 추가 (OCR)**")
    uploaded_file = st.file_uploader("문제 이미지 업로드", type=["jpg", "jpeg", "png"], key="ocr_upload")
    if uploaded_file and st.button("🚀 이미지 분석 및 추가"):
        with st.spinner('AI가 이미지를 읽고 번역 중...'):
            try:
                image = Image.open(uploaded_file)
                prompt = "이 이미지의 영어 문제를 텍스트로 추출하고 한국어로 번역해줘. '영어원문: ... / 한글해석: ...' 형식으로 답변해줘."
                response = gemini_model.generate_content([prompt, image])
                
                # 결과 저장
                st.session_state.master_problems.append({
                    'en': "이미지에서 추출된 원문",
                    'ko': response.text,
                    'type': '📸 OCR'
                })
                st.success("OCR 문제가 추가되었습니다!")
            except Exception as e:
                st.error(f"이미지 인식 실패: {e}")

with col2:
    st.markdown("#### **📚 데이터셋에서 추가**")
    if st.button("🔄 15문제 랜덤 배정 및 즉시 번역", type="primary"):
        if df is not None:
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
            pool = df[prob_col].dropna().unique().tolist()
            selected_en = random.sample(pool, 15) if len(pool) >= 15 else pool
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, en_text in enumerate(selected_en):
                status_text.text(f"⏳ {i+1}/{len(selected_en)} 번역 중...")
                try:
                    prompt = f"다음 문제를 한글로 자연스럽게 번역해줘. 번역문만 출력: {en_text}"
                    response = gemini_model.generate_content(prompt)
                    st.session_state.master_problems.append({
                        'en': en_text,
                        'ko': response.text,
                        'type': '📖 데이터셋'
                    })
                except:
                    continue
                progress_bar.progress((i + 1) / len(selected_en))
            
            status_text.empty()
            progress_bar.empty()
            st.success("15문제가 추가되었습니다!")
        else:
            st.error("데이터 파일을 찾을 수 없습니다.")

st.divider()

# --- 하단: 통합 문제 출력 섹션 ---
if st.session_state.master_problems:
    st.subheader(f"📝 현재 학습 리스트 ({len(st.session_state.master_problems)}문항)")
    
    for i, item in enumerate(st.session_state.master_problems, 1):
        with st.container():
            st.markdown(f"**Question {i}** [{item['type']}]")
            
            # 한글 번역본 즉시 노출
            st.info(item['ko'])
            
            # 영어 원문은 선택적으로 확인
            with st.expander("영어 원문 보기"):
                st.write(item['en'])
            
            st.text_input("정답 입력", key=f"ans_{i}")
            st.write("")
            st.divider()

    if st.button("🧨 리스트 전체 초기화"):
        st.session_state.master_problems = []
        st.rerun()
else:
    st.info("위의 메뉴를 통해 학습할 문제를 추가해 주세요.")

st.caption("LG AI CAMP NEW | Gemini 2.5 Flash 통합 OCR & 번역 엔진")
