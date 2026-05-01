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
    # 사용자의 요청에 따라 gemini-2.5-flash 모델명 지정
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

# 4. 세션 상태 관리 (배정된 문제 저장용)
if 'assigned_problems' not in st.session_state:
    st.session_state.assigned_problems = None

# 5. 메인 화면 구성
st.title("📖 AI 맞춤형 문제 은행 학습")
st.write("데이터셋에서 중복 없는 15개의 문제를 자동으로 배정받아 학습합니다.")
st.divider()

# 6. 문제 배정 버튼
if st.button("🔄 새로운 문제 15개 배정받기", type="primary"):
    if df is not None:
        # 데이터셋 내 문제 컬럼 식별
        prob_col = 'generated_problem_english' if 'generated_problem_english' in df.columns else 'problem_id'
        
        if prob_col in df.columns:
            # 중복 없는 고유 문제 리스트 추출
            problem_pool = df[prob_col].dropna().unique().tolist()
            
            if len(problem_pool) >= 15:
                selected = random.sample(problem_pool, 15)
                # 문제 정보 구조화
                st.session_state.assigned_problems = [
                    {'en': p, 'ko': '', 'translated': False} for p in selected
                ]
                st.success("✅ 15개의 문제가 새롭게 배정되었습니다!")
            else:
                st.warning(f"데이터가 부족하여 {len(problem_pool)}개만 불러왔습니다.")
        else:
            st.error("데이터셋에서 문제 컬럼을 찾을 수 없습니다.")
    else:
        st.error("데이터 파일(`bkt_training_dataset_english_problem.csv`)이 업로드되지 않았습니다.")

# 7. 배정된 문제 출력 및 개별 번역
if st.session_state.assigned_problems:
    st.subheader("📝 오늘의 학습 리스트")
    
    for i, prob_item in enumerate(st.session_state.assigned_problems):
        # 각 문제를 Expander(접이식 창)로 구성
        status_label = "✅ 완료" if prob_item['translated'] else "⏳ 대기"
        with st.expander(f"문제 {i+1} : {status_label}", expanded=not prob_item['translated']):
            
            # 영어 지문 표시
            st.info(f"**[English Question]**\n\n{prob_item['en']}")
            
            # 번역 및 해설 버튼
            if st.button(f"🌐 한글 번역 및 해설 보기 (Q{i+1})", key=f"btn_{i}"):
                with st.spinner('AI 튜터가 분석 중...'):
                    try:
                        # Gemini AI 호출 (2.5 Flash)
                        prompt = f"다음 영어 문제를 한국어로 번역하고, 핵심 포인트 하나를 설명해줘:\n\n{prob_item['en']}"
                        response = gemini_model.generate_content(prompt)
                        
                        # 결과 저장 및 상태 업데이트
                        st.session_state.assigned_problems[i]['ko'] = response.text
                        st.session_state.assigned_problems[i]['translated'] = True
                        st.rerun() # 화면 갱신
                    except Exception as e:
                        st.error(f"번역 도중 오류가 발생했습니다: {e}")
            
            # 번역 결과가 존재하는 경우에만 표시
            if prob_item['translated']:
                st.markdown("---")
                st.success(f"**[AI 해석 가이드]**\n\n{prob_item['ko']}")
                st.text_input("✍️ 정답을 입력하세요", key=f"ans_input_{i}")

    st.divider()
    if st.button("🎉 전체 학습 완료"):
        st.balloons()
        st.success("오늘 배정된 문제를 모두 확인하셨습니다!")

else:
    st.info("위 버튼을 눌러 학습할 문제를 배정받으세요.")

# 하단 정보
st.caption("LG AI CAMP NEW | Gemini 2.5 Flash Engine | Automated Learning System")
