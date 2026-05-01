import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import nest_asyncio
import random

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 영어 학습 튜터", page_icon="📝", layout="centered")

# 2. 보안 설정: Secrets로부터 Gemini API 키 로드
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # 가장 빠르고 효율적인 gemini-1.5-flash 모델 사용
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("❌ API 키를 찾을 수 없습니다. Streamlit Cloud의 Settings > Secrets를 확인해주세요.")
    st.stop()

# 3. 데이터 로드 (문제 은행 불러오기)
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

# 4. 메인 화면 구성
st.title("📖 AI 맞춤형 영어 문제장")
st.write("학습을 시작하려면 아래 버튼을 눌러주세요. 중복 없는 15문제가 제공됩니다.")
st.divider()

# 5. 세션 상태 관리 (문제 세트 유지)
if 'current_problems' not in st.session_state:
    st.session_state.current_problems = None

# 문제 생성 버튼
if st.button("🔄 새로운 문제 15개 생성하기", type="primary"):
    if df is not None:
        # 문제 데이터가 들어있는 컬럼 확인 (사용자별 데이터가 아닌 전체 문제 풀에서 추출)
        prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
        
        if prob_col in df.columns:
            # 전체 문제 풀에서 중복 제거 후 리스트화
            problem_pool = df[prob_col].dropna().unique().tolist()
            
            if len(problem_pool) >= 15:
                st.session_state.current_problems = random.sample(problem_pool, 15)
                st.success("✅ 새로운 15문제가 준비되었습니다!")
            else:
                st.session_state.current_problems = problem_pool
                st.warning(f"전체 문제가 부족하여 {len(problem_pool)}개만 불러왔습니다.")
        else:
            st.error("데이터셋에서 문제 내용을 찾을 수 없습니다.")
    else:
        st.error("데이터 파일을 불러올 수 없습니다. 파일명을 확인해주세요.")

# 6. 문제 출력 및 실시간 번역
if st.session_state.current_problems:
    for i, prob in enumerate(st.session_state.current_problems, 1):
        with st.container():
            st.markdown(f"### **Question {i}**")
            # 영어 지문을 눈에 잘 띄게 박스로 처리
            st.info(prob)
            
            # 번역 및 해설 버튼 (Gemini 활용)
            if st.button(f"🔍 번역 및 문법 설명 보기 (Q{i})", key=f"trans_{i}"):
                with st.spinner('AI 튜터가 해석 중...'):
                    try:
                        # Gemini에게 전달할 프롬프트
                        prompt = f"다음 영어 문제를 한국어로 번역하고, 핵심 단어와 문법을 2줄로 설명해줘:\n\n{prob}"
                        response = gemini_model.generate_content(prompt)
                        st.markdown("---")
                        st.markdown(f"**💡 AI 해석 및 가이드:**\n\n{response.text}")
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")
            
            # 정답 입력칸
            st.text_input("정답을 입력하세요", key=f"input_{i}")
            st.write("") # 문제 사이 간격
    
    st.divider()
    if st.button("🎉 학습 완료 (제출)"):
        st.balloons()
        st.success("오늘의 15문제를 모두 확인하셨습니다! 수고하셨습니다.")

else:
    st.info("좌측 상단의 버튼을 눌러 오늘의 학습을 시작하세요.")

# 하단 정보
st.caption("LG AI CAMP NEW | Gemini 1.5 Flash Engine")
