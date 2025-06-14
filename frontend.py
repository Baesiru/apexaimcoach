import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🎯 Apex Legends Aim Coach")
st.markdown("플레이 영상을 분석하여 최적의 마우스 감도를 추천합니다. **연발 총기**를 사용한 **1~5분 내외의 교전 영상**을 업로드해주세요.")

with st.sidebar:
    st.header("⚙️ 분석 설정")
    user_dpi = st.number_input("마우스 DPI (예: 800)", min_value=100, max_value=3200, value=800, step=50)
    user_sens = st.number_input("인게임 감도 (예: 1.2)", min_value=0.1, max_value=10.0, value=1.2, step=0.1)
    analysis_method = st.selectbox(
        "분석 모델 선택",
        ("LSTM 딥러닝 (추천)", "규칙 기반")  # 규칙 기반도 JSON을 반환하도록 수정해야 함
    )
    allowed_types = ["mp4", "mov", "avi", "mkv", "wmv", "webm"]
    uploaded_file = st.file_uploader(
        f"분석할 영상 파일을 업로드하세요. (지원 포맷: {', '.join(allowed_types)})",
        type=allowed_types
    )
    analyze_button = st.button("🚀 에임 분석 시작!")

if analyze_button:
    if uploaded_file:
        with st.spinner('영상을 분석 중입니다... 잠시만 기다려주세요. (1~5분 소요)'):
            try:
                # API 요청
                endpoint = "/analyze/lstm" if "LSTM" in analysis_method else "/analyze/rule-based"
                files = {'video_file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {'user_dpi': user_dpi, 'user_sens': user_sens}
                response = requests.post(f"{API_URL}{endpoint}", files=files, data=data, timeout=600)

                if response.status_code == 200:
                    st.success("✅ 분석이 완료되었습니다!")
                    res = response.json().get('result', {})

                    pred_class = res.get('summary', {}).get('predicted_class_str', 'N/A')
                    confidence = res.get('summary', {}).get('confidence', 0) * 100

                    st.subheader("📊 종합 분석 결과")

                    color = "red" if pred_class == "높음" else "blue" if pred_class == "낮음" else "green"
                    st.markdown(
                        f"분석 결과, 현재 감도는 **<span style='color:{color}; font-size: 24px;'>{pred_class}</span>** 경향을 보입니다.",
                        unsafe_allow_html=True)
                    st.progress(int(confidence))
                    st.caption(f"분석 신뢰도: {confidence:.2f}%")

                    st.info(res.get('recommendation', {}).get('detail_text', ''))

                    st.subheader("⚙️ 감도 설정 비교")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="**현재 eDPI**", value=f"{res.get('current_settings', {}).get('edpi', 0):.0f}")
                        st.caption(
                            f"DPI: {res.get('current_settings', {}).get('dpi', 0)} / 감도: {res.get('current_settings', {}).get('sens', 0):.3f}")

                    with col2:
                        st.metric(label="**추천 eDPI**",
                                  value=f"{res.get('recommended_settings', {}).get('edpi', 0):.0f}",
                                  delta=f"{(res.get('recommended_settings', {}).get('edpi', 0) - res.get('current_settings', {}).get('edpi', 0)):.0f}")
                        st.caption(f"추천 감도: {res.get('recommended_settings', {}).get('sens', 0):.3f}")

                    st.subheader("📄 상세 정보")

                    current_settings = res.get('current_settings', {})
                    recommended_settings = res.get('recommended_settings', {})

                    df = pd.DataFrame({
                        '항목': ['DPI', '인게임 감도', 'eDPI (유효 DPI)'],
                        '현재 설정': [
                            str(current_settings.get('dpi', 'N/A')),
                            f"{current_settings.get('sens', 0):.3f}",
                            f"{current_settings.get('edpi', 0):.0f}"
                        ],
                        '추천 설정': [
                            '- (유지)',
                            f"{recommended_settings.get('sens', 0):.3f}",
                            f"{recommended_settings.get('edpi', 0):.0f}"
                        ]
                    })
                    st.table(df.set_index('항목'))

                    st.subheader("🎯 감도 지표")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=res.get('recommended_settings', {}).get('edpi', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "eDPI 조정 추천", 'font': {'size': 20}},
                        delta={'reference': res.get('current_settings', {}).get('edpi', 0),
                               'increasing': {'color': "RebeccaPurple"}, 'decreasing': {'color': "MediumSeaGreen"}},
                        gauge={
                            'axis': {'range': [None, res.get('current_settings', {}).get('edpi', 0) * 1.5],
                                     'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, res.get('current_settings', {}).get('edpi', 0) * 0.8],
                                 'color': 'lightblue'},
                                {'range': [res.get('current_settings', {}).get('edpi', 0) * 0.8,
                                           res.get('current_settings', {}).get('edpi', 0) * 1.2],
                                 'color': 'lightgreen'}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': res.get('current_settings', {}).get('edpi', 0)}}))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)


                else:
                    error_detail = response.json().get('detail', response.text)
                    st.error(f"분석 중 오류가 발생했습니다 (오류 코드: {response.status_code}).\n\n오류 내용: {error_detail}")

            except requests.exceptions.RequestException as e:
                st.error(f"API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.\n\n오류 내용: {e}")
    elif analyze_button:
        st.warning("분석을 시작하려면 먼저 영상 파일을 업로드해주세요.")