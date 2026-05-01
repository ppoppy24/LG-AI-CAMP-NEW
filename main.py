import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import nest_asyncio
import random

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 자동 문제 배정 시스템", page_icon="📖", layout="centered")

# 2. 보안 설정 및 모델 로드 (Gemini 2.5 Flash)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ API 키를 설정해주세요. (Streamlit Secrets)")
    st.stop()

# 3. 데이터 로드 (문제 은행)
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

# 4. 세션 상태 관리 (배정된 문제 저장)
if 'assigned_problems' not in st.session_state:
    st.session_state.assigned_problems = None

# 5. 메인 화면 구성
st.title("📖 AI 맞춤형 문제 은행 학습")
st.write("데이터셋에서 엄선된 15개의 문제를 자동으로 배정합니다.")
st.divider()

# 6. 문제 배정 로직
if st.button("🔄 새로운 문제 15개 배정받기", type="primary"):
    if df is not None:
        # 데이터셋에서 문제 컬럼 찾기
        prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
        
        if prob_col in df.columns:
            # 중복 없는 고유 문제 리스트 생성
            problem_pool = df[prob_col].dropna().unique().tolist()
            
            if len(problem_pool) >= 15:
                # 15개 랜덤 추출하여 세션에 저장
                selected = random.sample(problem_pool, 15)
                st.session_state.assigned_problems = [
                    {'en': p, 'ko': '', 'translated': False} for p in selected
                ]
                st.success("✅ 15개의 문제가 새롭게 배정되었습니다!")
            else:
                st.warning(f"데이터가 부족하여 {len(problem_pool)}개만 불러왔습니다.")
        else:
            st.error("데이터셋에서 문제 컬럼을 찾을 수 없습니다.")
    else:
        st.error("데이터 파일(`bkt_training_dataset_english_problem.csv`)을 찾을 수 없습니다.")

# 7. 배정된 문제 출력 및 학습
if st.session_state.assigned_problems:
    st.subheader("📝 오늘의 학습 리스트")
    
    for i, prob_item in enumerate(st.session_state.assigned_problems):
        with st.expander(f"문제 {i+1} : {'✅ 번역 완료' if prob_item['translated'] else '🔍 미확인'}", expanded=not prob_item['translated']):
            # 영어 원문 표시
            st.info(f"**[English Question]**\n\n{prob_item['en']}")
            
            # AI 번역 및 해설 버튼
            if st.button(f"🌐 Gemini AI 한글 번역 보기 (Q{i+1})", key=f"trans_btn_{i}"):
                with st.spinner('AI 튜터가 해석 중...'):
                    try:
                        prompt = f"다음 영어 문제를 한국어로 친절하게 번역해주고, 핵심 포인트 하나만 알려줘:\n\n{prob_item['en']}"
                        response = gemini_model.generate_content(prompt)
                        st.session_state.assigned_problems[i]['ko'] = response.text
                        st.session_state.assigned_problems[i]['translated'] = True
                        st.rerun() # 상태 업데이트를 위해 재실행
            
            # 번역 결과가 있을 경우 표시
            if prob_item['translated']:
                st.success(f"**[한국어 해석 및 가이드]**\n\n{prob_item['ko']}")
                st.text_input("✍️ 정답을 입력하세요", key=f"answer_{i}")
                
    if st.button("🎉 오늘의 학습 완료하기"):
        st.balloons()
        st.success("모든 문제를 확인하셨습니다! 다음 학습을 위해 리셋하려면 위 버튼을 누르세요.")

else:
    st.info("좌측 상단의 [새로운 문제 15개 배정받기] 버튼을 눌러 학습을 시작하세요.")

# 하단 정보
st.caption("LG AI CAMP NEW | 데이터셋 자동 배정 & Gemini 2.5 Flash")
