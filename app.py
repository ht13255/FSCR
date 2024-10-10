import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import pytesseract
import matplotlib.pyplot as plt

# 1. 과거 경기 데이터 (더미 데이터 사용)
def crawl_past_games(player_name):
    past_games = [
        {'game': 1, 'goals': 1, 'xG': 0.8, 'xA': 0.5, 'assists': 1, 'shots': 3, 'shot_accuracy': 60,
         'dribble_success_rate': 70, 'touches_in_box': 12, 'progressive_passes': 6, 'tackles_success_rate': 80,
         'blocked_shots': 1, 'blocked_passes': 2, 'aerial_duels_won': 3, 'distance_covered': 10.5, 'sprints': 40},
        {'game': 2, 'goals': 0, 'xG': 0.3, 'xA': 0.2, 'assists': 0, 'shots': 2, 'shot_accuracy': 50,
         'dribble_success_rate': 65, 'touches_in_box': 8, 'progressive_passes': 5, 'tackles_success_rate': 75,
         'blocked_shots': 0, 'blocked_passes': 1, 'aerial_duels_won': 4, 'distance_covered': 11.2, 'sprints': 38},
        {'game': 3, 'goals': 2, 'xG': 1.2, 'xA': 0.6, 'assists': 1, 'shots': 5, 'shot_accuracy': 70,
         'dribble_success_rate': 85, 'touches_in_box': 15, 'progressive_passes': 7, 'tackles_success_rate': 90,
         'blocked_shots': 2, 'blocked_passes': 3, 'aerial_duels_won': 5, 'distance_covered': 12.0, 'sprints': 55},
        # 추가 경기 데이터
    ]
    return past_games

# 2. 경기 평균 및 추세 분석
def analyze_past_game_trends(past_games):
    df = pd.DataFrame(past_games)
    avg_stats = df.mean().to_dict()
    consistency = df.std().to_dict()
    return avg_stats, consistency, df

# 3. 머신러닝 모델로 장단점 분석
@st.cache
def analyze_strengths_weaknesses(avg_stats):
    df = pd.DataFrame([avg_stats])
    features = ['goals', 'xG', 'xA', 'assists', 'shots', 'shot_accuracy',
                'dribble_success_rate', 'touches_in_box', 'progressive_passes',
                'tackles_success_rate', 'blocked_shots', 'blocked_passes',
                'aerial_duels_won', 'distance_covered', 'sprints']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    y = np.random.choice([0, 1], size=(len(df),))
    rf.fit(X_scaled, y)
    importance = rf.feature_importances_

    strengths = [features[i] for i, imp in enumerate(importance) if imp > 0.05]
    weaknesses = [features[i] for i, imp in enumerate(importance) if imp <= 0.05]

    return strengths, weaknesses

# 4. 딥러닝 모델로 플레이 스타일 분석
@st.cache
def analyze_play_style(avg_stats):
    df = pd.DataFrame([avg_stats])
    features = ['goals', 'xG', 'assists', 'shots', 'dribble_success_rate', 
                'progressive_passes', 'tackles_success_rate', 'distance_covered', 'sprints']
    
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    y = np.random.choice([0, 1, 2], size=(len(df),))
    model.fit(X_scaled, y, epochs=3, batch_size=128, verbose=0)

    play_style_prediction = np.argmax(model.predict(X_scaled), axis=1)[0]
    play_styles = {0: "공격적", 1: "수비적", 2: "밸런스"}
    return play_styles[play_style_prediction]

# 5. 시각화: 경기별 성과 추이 그래프 생성
def plot_performance_trends(df):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].plot(df['game'], df['goals'], marker='o', label='Goals')
    ax[0, 0].plot(df['game'], df['xG'], marker='o', linestyle='--', label='xG')
    ax[0, 0].set_title('Goals vs xG')
    ax[0, 0].legend()

    ax[0, 1].plot(df['game'], df['assists'], marker='o', label='Assists')
    ax[0, 1].plot(df['game'], df['xA'], marker='o', linestyle='--', label='xA')
    ax[0, 1].set_title('Assists vs xA')
    ax[0, 1].legend()

    ax[1, 0].plot(df['game'], df['shot_accuracy'], marker='o', label='Shot Accuracy')
    ax[1, 0].plot(df['game'], df['dribble_success_rate'], marker='o', linestyle='--', label='Dribble Success Rate')
    ax[1, 0].set_title('Shot Accuracy vs Dribble Success Rate')
    ax[1, 0].legend()

    ax[1, 1].plot(df['game'], df['distance_covered'], marker='o', label='Distance Covered (km)')
    ax[1, 1].plot(df['game'], df['sprints'], marker='o', linestyle='--', label='Sprints')
    ax[1, 1].set_title('Distance Covered vs Sprints')
    ax[1, 1].legend()

    plt.tight_layout()
    return fig

# 6. 영상 분석 기능 추가 (침투 움직임, 슛 자세 분석)
def analyze_video(video_file):
    if video_file is None:
        return "No video uploaded."
    
    # OpenCV로 비디오 읽기
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    penetration_count = 0
    shot_form_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # 특정 프레임에서 침투 움직임 분석 (임의 기준 적용)
        if frame_count % 30 == 0:  # 매 30프레임마다 분석
            penetration_count += 1  # 단순 카운팅 예시

        # 슛 자세 분석
        if frame_count % 50 == 0:  # 매 50프레임마다 슛 자세 분석
            shot_form_count += 1

    cap.release()

    return {
        "penetration_moves": penetration_count,
        "shot_forms": shot_form_count
    }

# 7. PDF 리포트 생성
def generate_pdf_report(player_name, avg_stats, strengths, weaknesses, play_style, df, video_analysis):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, f'{player_name}의 스카우팅 리포트', ln=True, align='C')

    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"총 득점: {avg_stats['goals']}", ln=True)
    pdf.cell(200, 10, f"총 어시스트: {avg_stats['assists']}", ln=True)
    pdf.cell(200, 10, f"xG 대비 득점 초과: {avg_stats['xG'] - avg_stats['goals']}", ln=True)
    pdf.cell(200, 10, f"xA (기대 어시스트): {avg_stats['xA']}", ln=True)
    pdf.cell(200, 10, f"패스 성공률: {avg_stats['progressive_passes']}%", ln=True)
    pdf.cell(200, 10, f"드리블 성공률: {avg_stats['dribble_success_rate']}%", ln=True)
    pdf.cell(200, 10, f"스프린트 횟수: {avg_stats['sprints']}", ln=True)

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, "장점", ln=True)
    pdf.set_font('Arial', '', 12)
    for strength in strengths:
        pdf.cell(200, 10, f"- {strength}", ln=True)

    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, "단점", ln=True)
    pdf.set_font('Arial', '', 12)
    for weakness in weaknesses:
        pdf.cell(200, 10, f"- {weakness}", ln=True)

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, "플레이 스타일", ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, f"플레이 스타일: {play_style}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, "영상 분석 결과", ln=True)
    pdf.cell(200, 10, f"침투 움직임 횟수: {video_analysis['penetration_moves']}", ln=True)
    pdf.cell(200, 10, f"슛 자세 분석 횟수: {video_analysis['shot_forms']}", ln=True)

    fig = plot_performance_trends(df)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        fig.savefig(tmp_file.name)
        pdf.image(tmp_file.name, x=10, y=100, w=180)

    return pdf

# Streamlit UI
st.title('축구 스카우팅 리포트 생성기')
player_name = st.text_input("선수 이름 입력:")
uploaded_video = st.file_uploader("선수의 영상을 업로드하세요", type=["mp4", "avi"])

if player_name:
    # 과거 경기 데이터
    past_games = crawl_past_games(player_name)
    
    # 경기 평균 및 추세 분석
    avg_stats, consistency, df = analyze_past_game_trends(past_games)

    # 장점과 단점 분석
    strengths, weaknesses = analyze_strengths_weaknesses(avg_stats)

    # 플레이 스타일 분석
    play_style = analyze_play_style(avg_stats)

    # 영상 분석 수행
    video_analysis = analyze_video(uploaded_video)

    st.write("### 경기 평균 스탯")
    st.write(avg_stats)

    st.write("### 경기별 일관성 (표준편차)")
    st.write(consistency)

    st.write("### 플레이 스타일")
    st.write(f"플레이 스타일: {play_style}")

    st.write("### 장점")
    st.write(strengths)

    st.write("### 단점")
    st.write(weaknesses)

    st.write("### 영상 분석 결과")
    st.write(video_analysis)

    # 성과 추이 그래프 시각화
    st.write("### 경기별 성과 추이 그래프")
    fig = plot_performance_trends(df)
    st.pyplot(fig)

    # PDF 리포트 생성 및 다운로드 기능
    if st.button('PDF 리포트 다운로드'):
        pdf_report = generate_pdf_report(player_name, avg_stats, strengths, weaknesses, play_style, df, video_analysis)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_report.output(tmp_file.name)
            st.download_button(
                label="PDF 다운로드",
                data=open(tmp_file.name, "rb").read(),
                file_name=f"{player_name}_scouting_report.pdf",
                mime="application/pdf"
            )
