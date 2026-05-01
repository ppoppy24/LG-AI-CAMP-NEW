import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
import random
import re
import cv2
import easyocr
import joblib
import os
from base64 import b64decode
from IPython.display import display, Javascript
from google.colab.output import eval_js
from google import genai

# 1. 환경 설정
nest_asyncio.apply()
reader = easyocr.Reader(['ko', 'en'])

# API 및 모델 설정
API_KEY = "AIzaSyDE9pzlh_JR9WvuxGbI0C2OzG36dC-r7Wg"
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# 🔥 경로 설정: 왼쪽 폴더 메뉴에서 파일 우클릭 -> '경로 복사'한 값을 여기에 넣으세요.
RF_MODEL_PATH = '/content/models/bkt_rf_model.pkl' 

# ===============================
# 2. RF 모델 진단 로직 (variance 추가 버전)
# ===============================
def diagnose_learning_status(results):
    if not os.path.exists(RF_MODEL_PATH):
        return "SYSTEM_ERROR_MODEL_NOT_FOUND", 0, 0, 0

    try:
        model = joblib.load(RF_MODEL_PATH)
        
        # 피처 계산
        correctness = [r['is_correct'] for r in results]
        accuracy = sum(correctness) / len(results) if results else 0
        initial_k = results[0]['is_correct'] * 0.4 if results else 0
        final_k = results[-1]['is_correct'] * 0.8 if results else 0
        learning_gain = max(0, final_k - initial_k)
        
        # 🔥 에러 해결의 핵심: variance(분산) 피처 추가
        variance = np.var(correctness) if len(correctness) > 1 else 0.0

        # 모델 학습 시와 동일한 컬럼명과 순서로 구성
        input_df = pd.DataFrame([[initial_k, final_k, learning_gain, accuracy, variance]], 
                                columns=['initial_knowledge', 'final_knowledge', 'learning_gain', 'accuracy', 'variance'])
        
        status = model.predict(input_df)[0]
        return status, accuracy, learning_gain, initial_k
    except Exception as e:
        return f"SYSTEM_ERROR_LOAD_FAILED_{str(e)}", 0, 0, 0

# ===============================
# 3. 유틸리티 함수 (기존 동일)
# ===============================
def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''async function takePhoto(quality) { const div = document.createElement('div'); const btn = document.createElement('button'); btn.textContent = '📷 촬영'; div.appendChild(btn); const video = document.createElement('video'); const stream = await navigator.mediaDevices.getUserMedia({video: true}); document.body.appendChild(div); div.appendChild(video); video.srcObject = stream; await video.play(); await new Promise((resolve) => btn.onclick = resolve); const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight; canvas.getContext('2d').drawImage(video, 0, 0); stream.getTracks().forEach(track => track.stop()); div.remove(); return canvas.toDataURL('image/jpeg', quality); }''')
    display(js); data = eval_js('takePhoto({})'.format(quality)); binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f: f.write(binary)
    return filename

def ocr_read(path):
    img = cv2.imread(path)
    if img is None: return ""
    img = cv2.resize(img, None, fx=2, fy=2); result = reader.readtext(img)
    return " ".join([r[1] for r in result])

def extract_number(text):
    nums = re.findall(r"\d+", str(text))
    return int(nums[-1]) if nums else None

def prime_factorization(n):
    factors = []; d = 2
    while d*d <= n:
        while n % d == 0: factors.append(d); n //= d
        d += 1
    if n > 1: factors.append(n)
    return factors

def normalize(text):
    text = str(text).lower().replace("*","x").replace("×","x").replace(" ","")
    try: return sorted([int(x) for x in text.split("x") if x.isdigit()])
    except: return []

def generate_recommend_num(original_num):
    factors = prime_factorization(original_num); primes = [2, 3, 5, 7]
    if len(factors) > 2 and random.choice([True, False]): factors.pop(random.randint(0, len(factors)-1))
    else: factors.append(random.choice(primes))
    new_num = 1
    for f in factors: new_num *= f
    return new_num if new_num > 1 else original_num + 2

def get_user_answer(idx, label="기본"):
    print(f"\n[{label} Q{idx}] 1: 타이핑, 2: 사진"); choice = input("선택: ").strip()
    if choice == '2':
        filename = take_photo(f"ans_{label}_{idx}.jpg"); ocr_text = ocr_read(filename)
        print(f"👉 인식 결과: {ocr_text}"); fix = input("수정(맞으면 엔터): ").strip()
        return fix if fix != "" else ocr_text
    return input("✏️ 답안 입력: ").strip()

# ===============================
# 4. 메인 실행 함수
# ===============================
async def run_study_system():
    file_path = "/content/data/bkt_training_dataset_english_problem.csv"
    try: df = pd.read_csv(file_path)
    except: print("데이터셋 로드 실패."); return

    sample = df.drop_duplicates(subset=['generated_problem_english']).sample(n=min(5, len(df))).reset_index(drop=True)
    results = []
    
    print(f"\n🚀 [1단계] 기초 문제 풀이 시작")
    for i, row in sample.iterrows():
        num = extract_number(row["generated_problem_english"])
        print(f"\nQ{i+1}. {num}을 소인수분해하세요.")
        u_ans = get_user_answer(i+1, "기본")
        correct_list = prime_factorization(num)
        is_correct = 1 if normalize(u_ans) == sorted(correct_list) else 0
        results.append({"num": num, "is_correct": is_correct, "user_ans": u_ans, "correct_ans": 'x'.join(map(str, correct_list))})
        print("✅ 정답!" if is_correct else f"❌ 오답! (정답: {'x'.join(map(str, correct_list))})")

    wrong_list = [r for r in results if r["is_correct"] == 0]
    
    # 기초 문제를 하나라도 틀린 경우에만 피드백과 추천문제가 나옵니다.
    if wrong_list:
        print("\n" + "="*40 + "\n🤖 1차 AI 오답 분석 및 격려")
        summary = "\n".join([f"문제: {r['num']}, 학생답: {r['user_ans']}, 정답: {r['correct_ans']}" for r in wrong_list])
        prompt1 = f"수학 선생님으로서 다음 오답들을 분석하고 정답 여부를 알려주며 격려해주세요.\n{summary}"
        try:
            resp1 = await client.aio.models.generate_content(model=MODEL_NAME, contents=prompt1)
            print(resp1.text.strip())
        except: print("1차 피드백 생성 중...")

        print("\n" + "💡" * 15 + "\n🛠️ 맞춤 클리닉 추천 문제")
        recommend_results = []
        for j, wr in enumerate(wrong_list):
            new_num = generate_recommend_num(wr["num"])
            print(f"\n[추천 Q{j+1}] {new_num}을 소인수분해하세요.")
            u_ans = get_user_answer(j+1, "추천")
            correct_list = prime_factorization(new_num)
            is_correct = 1 if normalize(u_ans) == sorted(correct_list) else 0
            
            # 추천 문제 정답 여부 즉시 출력
            if is_correct: print("✅ 정답입니다!")
            else: print(f"❌ 오답입니다. 정답은 {'x'.join(map(str, correct_list))}입니다.")
            
            recommend_results.append({"num": new_num, "is_correct": is_correct, "user_ans": u_ans, "correct_ans": 'x'.join(map(str, correct_list))})

        # 최종 RF 모델 진단
        status, acc, gain, init = diagnose_learning_status(results + recommend_results)

        if "SYSTEM_ERROR" in str(status):
            print(f"\n⚠️ 모델 진단 에러: {status}")
        else:
            print("\n" + "="*40 + "\n🎓 AI 선생님의 최종 진단 가이드")
            rec_summary = "\n".join([f"- 숫자 {r['num']}: {'정답' if r['is_correct'] else '오답'}" for r in recommend_results])
            prompt2 = f"""
            수학 교육 전문가로서 아래 데이터를 분석하세요.
            이미 모델이 분석한 학생의 상태는 '{status}'입니다.
            
            [진단 데이터]
            - 상태 분류: {status}
            - 전체 정확도: {acc*100:.1f}%
            - 학습 성장도: {gain:.2f}
            
            [추천 문제 결과]
            {rec_summary}
            
            지침: 
            1. 학생의 상태가 '{status}'(Learning 또는 Guessing)임을 명확히 언급하며 조언을 시작하세요.
            2. "모델이 없다"거나 "데이터가 부족하다"는 말은 절대 하지 마세요. 
            3. 위 수치를 근거로 아주 구체적인 학습 방향을 제시하세요.
            """
            try:
                resp2 = await client.aio.models.generate_content(model=MODEL_NAME, contents=prompt2)
                print(resp2.text.strip())
            except: print("최종 가이드 생성 실패")
    else:
        print("\n🎉 모든 기초 문제를 맞혔습니다! 소인수분해 마스터입니다.")

if __name__ == "__main__":
    asyncio.run(run_study_system())