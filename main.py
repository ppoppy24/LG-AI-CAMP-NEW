import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import nest_asyncio
import random

# 1. 초기 환경 설정
nest_asyncio.apply()
st.set_page_config(page_title="AI 실시간 학습 답안지", page_icon="✍️", layout="centered")

# 2. 보안 설정 및 모델 로드 (Gemini 2.5 Flash)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ API 키를 설정해주세요. (Streamlit Cloud의 Secrets 메뉴)")
    st.stop()

# 3. 데이터 로드 (학습 시스템 데이터셋)
@st.cache_resource
def load_data():
    # 사용자의 교육 시스템 데이터셋 로드
    data_path = 'bkt_training_dataset_english_problem.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

# 4. 세션 상태 관리 (문제와 입력한 답안 저장)
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None

# 5. 메인 화면 구성
st.title("✍️ AI 맞춤형 학습 답안지")
st.write("아래 버튼을 누르면 오늘의 15문제가 배정됩니다. 각 문제에 답을 입력하세요.")

# 6. 문제 배정 및 자동 번역 (최초 1회 실행)
if st.button("🏁 오늘의 문제 15개 받기", type="primary"):
    if df is not None:
        prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
        pool = df[prob_col].dropna().unique().tolist()
        
        if len(pool) >= 15:
            selected_en = random.sample(pool, 15)
            quiz_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, en_text in enumerate(selected_en):
                status_text.text(f"🤖 문제를 한글로 준비 중... ({i+1}/15)")
                try:
                    # 문제 즉시 한글화
                    prompt = f"다음 영어 문제를 한국어로 자연스럽게 번역해줘. 번역문만 출력: {en_text}"
                    response = gemini_model.generate_content(prompt)
                    quiz_list.append({
                        'en': en_text,
                        'ko': response.text,
                        'user_answer': ""
                    })
                except:
                    quiz_list.append({'en': en_text, 'ko': en_text, 'user_answer': ""})
                progress_bar.progress((i + 1) / 15)
            
            st.session_state.quiz_data = quiz_list
            status_text.empty()
            progress_bar.empty()
            st.rerun()
    else:
        st.error("데이터셋 파일을 찾을 수 없습니다.")

st.divider()

# 7. 답안 입력 섹션
if st.session_state.quiz_data:
    for i, item in enumerate(st.session_state.quiz_data):
        st.markdown(f"#### **Q{i+1}.**")
        st.info(item['ko']) # 번역된 문제 표시
        
        # 사용자가 답을 입력하는 곳 (핵심 기능)
        user_input = st.text_input(f"Q{i+1}번 답안 입력", key=f"input_{i}", placeholder="정답 또는 풀이 과정을 입력하세요.")
        st.session_state.quiz_data[i]['user_answer'] = user_input
        
        with st.expander("영어 원문 확인"):
            st.write(item['en'])
        st.write("")

    # 8. 최종 제출 및 AI 피드백
    if st.button("📤 답안 제출하고 채점받기"):
        st.divider()
        st.subheader("📊 AI 학습 진단 및 피드백") # 학습 진단 도구 활용
        
        with st.spinner('AI 튜터가 답안을 채점하고 피드백을 생성 중입니다...'):
            all_answers = ""
            for i, item in enumerate(st.session_state.quiz_data):
                all_answers += f"문제{i+1}: {item['en']}\n학생답변: {item['user_answer']}\n\n"
            
            # 채점 및 피드백 프롬프트
            feedback_prompt = f"다음은 학생이 푼 15개의 영어 문제와 답안이야. 각 문제의 정답 여부를 판단하고, 틀린 부분에 대해 친절하게 설명해줘:\n\n{all_answers}"
            feedback_res = gemini_model.generate_content(feedback_prompt)
            
            st.success("채점이 완료되었습니다!")
            st.markdown(feedback_res.text)
            st.balloons()

else:
    st.info("위의 버튼을 눌러 학습을 시작하세요.")

st.caption("LG AI CAMP NEW | Gemini 2.5 Flash 자동 채점 시스템")
