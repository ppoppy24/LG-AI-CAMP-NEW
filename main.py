import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import nest_asyncio
import random

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 맞춤형 문제 학습", page_icon="📖", layout="centered")

# 2. 보안 설정 및 모델 로드 (Gemini 2.0 Flash 적용)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # 404 에러 방지를 위해 최신 모델인 gemini-2.0-flash로 설정
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets를 확인해주세요.")
    st.stop()

# 3. 데이터 로드
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

# 4. 즉시 번역 함수
def translate_immediately(english_problems):
    translated_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🤖 Gemini AI가 문제를 한글로 번역하고 있습니다...")
    
    for i, prob in enumerate(english_problems):
        try:
            # 문제를 즉시 한글로 번역하도록 프롬프트 구성
            prompt = f"다음 영어 교육 문제를 한국어로 자연스럽게 번역해줘. 오직 번역된 문장만 출력해:\n\n{prob}"
            response = gemini_model.generate_content(prompt)
            translated_list.append({"en": prob, "ko": response.text})
        except Exception:
            # 오류 발생 시 원문 표시
            translated_list.append({"en": prob, "ko": "[번역 오류] " + prob})
        
        progress_bar.progress((i + 1) / len(english_problems))
    
    status_text.empty()
    progress_bar.empty()
    return translated_list

# 5. 메인 화면 구성
st.title("📖 AI 맞춤형 한글 문제장")
st.write("버튼을 누르면 15개의 문제를 AI가 즉시 한글로 번역하여 제공합니다.")
st.divider()

# 세션 상태 관리
if 'display_problems' not in st.session_state:
    st.session_state.display_problems = None

# 문제 생성 및 즉시 번역 버튼
if st.button("🔄 새로운 문제 15개 생성 (자동 번역)", type="primary"):
    if df is not None:
        prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
        
        if prob_col in df.columns:
            problem_pool = df[prob_col].dropna().unique().tolist()
            
            if len(problem_pool) >= 15:
                selected_en = random.sample(problem_pool, 15)
                # 생성과 동시에 번역 수행
                st.session_state.display_problems = translate_immediately(selected_en)
                st.success("✅ 번역이 완료되었습니다!")
            else:
                st.warning("데이터가 부족합니다.")
        else:
            st.error("문제 컬럼을 찾을 수 없습니다.")
    else:
        st.error("데이터 파일을 찾을 수 없습니다.")

# 6. 문제 출력 (처음부터 한글로 표시)
if st.session_state.display_problems:
    for i, item in enumerate(st.session_state.display_problems, 1):
        with st.container():
            st.markdown(f"### **문제 {i}**")
            # 한글 번역본을 메인으로 표시
            st.info(item['ko'])
            
            # 필요한 경우 영어 원문 확인 (접이식)
            with st.expander("영어 원문 보기"):
                st.text(item['en'])
            
            st.text_input("정답 입력", key=f"ans_{i}")
            st.write("")
            st.divider()
            
    if st.button("🎉 학습 완료"):
        st.balloons()
else:
    st.info("새로운 문제 생성 버튼을 눌러주세요.")

st.caption("LG AI CAMP NEW | Gemini 2.5 Flash Engine")
