import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import nest_asyncio
import random
from PIL import Image

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 영어 학습 & OCR", page_icon="📸", layout="centered")

# 2. 보안 설정 및 모델 로드 (Gemini 2.5 적용)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # 요청하신 대로 버전을 2.5로 변경했습니다.
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets를 확인해주세요.")
    st.stop()

# 3. 데이터 로드 (문제 은행)
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

# 4. 즉시 번역 함수
def translate_text(text):
    try:
        prompt = f"다음 교육 문제를 한국어로 자연스럽게 번역해줘. 오직 번역본만 출력해:\n\n{text}"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"번역 실패: {e}"

# 5. 메인 화면 구성
st.title("📸 AI OCR & 한글 문제 학습")
st.write("이미지를 업로드하여 문제를 인식하거나, 문제 은행에서 15문제를 가져올 수 있습니다.")

# 6. 탭 구성: OCR 기능과 문제 은행 분리
tab1, tab2 = st.tabs(["🔍 이미지 문제 인식 (OCR)", "📚 문제 은행 (15문항)"])

# --- 탭 1: OCR 기능 ---
with tab1:
    st.subheader("📷 문제 사진 업로드")
    uploaded_file = st.file_uploader("문제지가 찍힌 이미지를 업로드하세요 (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 이미지', use_container_width=True)
        
        if st.button("🚀 이미지에서 문제 추출 및 번역"):
            with st.spinner('Gemini 2.5가 이미지를 읽고 번역 중입니다...'):
                try:
                    # 이미지와 텍스트 프롬프트를 함께 전달 (멀티모달 OCR)
                    prompt = "이 이미지에 적힌 교육 문제를 텍스트로 추출하고, 그 내용을 한국어로 번역해서 보여줘. [원본 영어]와 [한국어 해석]으로 구분해줘."
                    response = gemini_model.generate_content([prompt, image])
                    
                    st.success("인식 완료!")
                    st.markdown("### 📝 분석 결과")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"OCR 처리 중 오류 발생: {e}")

# --- 탭 2: 문제 은행 (자동 번역 포함) ---
with tab2:
    if 'bank_problems' not in st.session_state:
        st.session_state.bank_problems = None

    if st.button("🔄 새로운 문제 15개 생성 및 자동 번역", type="primary"):
        if df is not None:
            prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
            if prob_col in df.columns:
                problem_pool = df[prob_col].dropna().unique().tolist()
                
                if len(problem_pool) >= 15:
                    selected_en = random.sample(problem_pool, 15)
                    translated_results = []
                    
                    progress_bar = st.progress(0)
                    for i, en_text in enumerate(selected_en):
                        ko_text = translate_text(en_text)
                        translated_results.append({"en": en_text, "ko": ko_text})
                        progress_bar.progress((i + 1) / 15)
                    
                    st.session_state.bank_problems = translated_results
                    st.success("✅ 15문제 번역 완료!")
            else:
                st.error("데이터셋 형식이 맞지 않습니다.")
        else:
            st.error("데이터 파일을 찾을 수 없습니다.")

    # 문제 출력 (처음부터 한글로 표시)
    if st.session_state.bank_problems:
        for i, item in enumerate(st.session_state.bank_problems, 1):
            st.markdown(f"#### **문제 {i}**")
            st.info(item['ko'])
            with st.expander("영어 원문 보기"):
                st.text(item['en'])
            st.text_input("정답 입력", key=f"bank_ans_{i}")
            st.divider()

st.caption("LG AI CAMP NEW | Gemini 2.5 Flash Engine")
