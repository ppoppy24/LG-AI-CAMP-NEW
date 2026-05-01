import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import nest_asyncio
import random

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 실시간 번역 학습기", page_icon="⚡", layout="centered")

# 2. 보안 설정 및 모델 로드 (Gemini 2.5 Flash)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ API 키를 설정해주세요. (Streamlit Cloud의 Secrets 메뉴)")
    st.stop()

# 3. 데이터 로드 (문제 은행)
@st.cache_resource
def load_data():
    data_path = 'bkt_training_dataset_english_problem.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

# 4. 세션 상태 관리
if 'assigned_problems' not in st.session_state:
    st.session_state.assigned_problems = None

# 5. 메인 화면 구성
st.title("⚡ AI 즉시 번역 문제장")
st.write("버튼을 누르면 15문제를 가져와서 즉시 한글로 번역합니다.")
st.divider()

# 6. 문제 배정 및 일괄 번역 로직
if st.button("🔄 새로운 문제 15개 배정 및 번역 시작", type="primary"):
    if df is not None:
        prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
        
        if prob_col in df.columns:
            problem_pool = df[prob_col].dropna().unique().tolist()
            
            if len(problem_pool) >= 15:
                selected_en = random.sample(problem_pool, 15)
                translated_list = []
                
                # 진행 상태 표시를 위한 UI
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, en_text in enumerate(selected_en):
                    status_text.text(f"⏳ {i+1}/15번째 문제 번역 중...")
                    try:
                        # 즉시 번역 수행
                        prompt = f"다음 영어 문제를 한국어로 자연스럽게 번역해줘. 번역문만 출력:\n\n{en_text}"
                        response = gemini_model.generate_content(prompt)
                        translated_list.append({
                            'en': en_text,
                            'ko': response.text,
                            'completed': True
                        })
                    except Exception:
                        translated_list.append({
                            'en': en_text,
                            'ko': "번역에 실패했습니다.",
                            'completed': False
                        })
                    # 진행 바 업데이트
                    progress_bar.progress((i + 1) / 15)
                
                st.session_state.assigned_problems = translated_list
                status_text.success("✅ 15문제 모두 번역 및 배정이 완료되었습니다!")
                progress_bar.empty()
            else:
                st.warning("데이터가 부족합니다.")
        else:
            st.error("컬럼을 찾을 수 없습니다.")
    else:
        st.error("데이터 파일을 찾을 수 없습니다.")

# 7. 배정된 문제 출력 (처음부터 번역본 표시)
if st.session_state.assigned_problems:
    st.subheader("📝 오늘의 한글 번역 문제 리스트")
    
    for i, item in enumerate(st.session_state.assigned_problems):
        with st.container():
            st.markdown(f"#### **Question {i+1}**")
            
            # 메인 화면에 한글 번역본을 바로 노출
            st.info(item['ko'])
            
            # 원문이 궁금할 때만 열어볼 수 있도록 배치
            with st.expander("영어 원문 확인"):
                st.write(item['en'])
            
            st.text_input("정답 입력", key=f"ans_{i}", placeholder="여기에 답을 입력하세요.")
            st.write("")
            st.divider()
            
    if st.button("🎉 학습 완료"):
        st.balloons()
else:
    st.info("버튼을 눌러 학습을 시작하세요.")

st.caption("LG AI CAMP NEW | Gemini 2.5 Flash | 자동 번역 시스템")
