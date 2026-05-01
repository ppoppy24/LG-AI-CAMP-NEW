import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import nest_asyncio
from PIL import Image
import io

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 맞춤형 문제 등록", page_icon="✍️", layout="centered")

# 2. 보안 설정 및 모델 로드 (Gemini 2.5 Flash)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ API 키를 설정해주세요.")
    st.stop()

# 3. 데이터 로드 (문제 은행 - 필요시 활용)
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    return pd.read_csv(data_path) if os.path.exists(data_path) else None

df = load_data()

# 4. 세션 상태 관리 (최대 15개의 슬롯 관리)
if 'study_slots' not in st.session_state:
    # 각 슬롯은 {type: 'none', en: '', ko: '', completed: False} 구조
    st.session_state.study_slots = [{'type': 'none', 'en': '', 'ko': '', 'completed': False} for _ in range(15)]

# 5. 메인 화면 구성
st.title("📝 나만의 AI 맞춤형 문제장")
st.write("각 문제 번호 아래의 입력 방식을 선택하여 문제를 등록하세요.")
st.divider()

# 6. 15개의 문제 슬롯 생성
for i in range(15):
    with st.expander(f"📍 문제 {i+1} : {'✅ 등록 완료' if st.session_state.study_slots[i]['completed'] else '⏳ 미등록'}", expanded=not st.session_state.study_slots[i]['completed']):
        
        # 입력 방식 선택 (라디오 버튼)
        input_type = st.radio(
            f"입력 방식 선택 (Q{i+1})",
            ["선택 안함", "📷 사진 찍기/업로드", "⌨️ 직접 타이핑"],
            key=f"type_{i}",
            horizontal=True
        )
        st.session_state.study_slots[i]['type'] = input_type

        # A. 사진 찍기 방식
        if input_type == "📷 사진 찍기/업로드":
            img_file = st.file_uploader(f"문제 사진을 올려주세요 (Q{i+1})", type=['jpg', 'jpeg', 'png'], key=f"img_{i}")
            if img_file and st.button(f"🚀 AI 분석 및 번역 실행 (Q{i+1})", key=f"btn_img_{i}"):
                with st.spinner('이미지 분석 중...'):
                    image = Image.open(img_file)
                    prompt = "이 이미지의 영어 문제를 추출하고 한국어로 번역해줘. 결과는 반드시 '영어: (내용) / 한글: (내용)' 형식으로 써줘."
                    response = gemini_model.generate_content([prompt, image])
                    
                    # 간단한 파싱 (응답 결과에 따라 조정 가능)
                    res_text = response.text
                    st.session_state.study_slots[i]['en'] = res_text # 원문+해석 전체 저장
                    st.session_state.study_slots[i]['completed'] = True
                    st.rerun()

        # B. 직접 타이핑 방식
        elif input_type == "⌨️ 직접 타이핑":
            en_input = st.text_area(f"영어 문제를 입력하세요 (Q{i+1})", key=f"text_{i}")
            if en_input and st.button(f"🌐 한글로 자동 번역 (Q{i+1})", key=f"btn_text_{i}"):
                with st.spinner('번역 중...'):
                    prompt = f"다음 영어 문제를 한국어로 자연스럽게 번역해줘:\n\n{en_input}"
                    response = gemini_model.generate_content(prompt)
                    st.session_state.study_slots[i]['en'] = en_input
                    st.session_state.study_slots[i]['ko'] = response.text
                    st.session_state.study_slots[i]['completed'] = True
                    st.rerun()

        # 등록된 내용 표시
        if st.session_state.study_slots[i]['completed']:
            st.markdown("---")
            st.success("📝 등록된 내용")
            if st.session_state.study_slots[i]['ko']: # 타이핑 시
                st.info(f"**[한글 해석]**\n{st.session_state.study_slots[i]['ko']}")
                with st.expander("영어 원문 보기"):
                    st.write(st.session_state.study_slots[i]['en'])
            else: # 이미지 분석 시
                st.write(st.session_state.study_slots[i]['en'])
            
            st.text_input("✍️ 정답을 입력하세요", key=f"final_ans_{i}")
            if st.button(f"🗑 다시 작성하기 (Q{i+1})", key=f"reset_{i}"):
                st.session_state.study_slots[i] = {'type': 'none', 'en': '', 'ko': '', 'completed': False}
                st.rerun()

# 7. 하단 초기화 버튼
st.divider()
if st.button("🧨 전체 문제장 초기화"):
    st.session_state.study_slots = [{'type': 'none', 'en': '', 'ko': '', 'completed': False} for _ in range(15)]
    st.rerun()

st.caption("LG AI CAMP NEW | Gemini 2.5 Flash 기반 맞춤형 학습장")
