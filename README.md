# **Apex Legends AI Aim Coach**

## 📖 프로젝트 개요

**Apex Legends AI Aim Coach**는 사용자의 플레이 영상을 분석하여 개인에게 최적화된 마우스 감도를 추천해주는 AI 기반 에임 코칭 프로그램입니다.

본 프로젝트는 컴퓨터 비전(YOLOv8, OpenCV) 기술을 사용해 영상에서 적의 위치와 조준점을 실시간으로 추적하고, 시계열 분석 딥러닝 모델(LSTM)을 통해 사용자의 에임 패턴(오버슈팅, 트래킹 랙 등)을 분석합니다. 분석 결과를 바탕으로 현재 감도가 높은지, 낮은지, 또는 적절한지를 판단하고 구체적인 감도 조절 가이드를 제공합니다.

분석 모델은 **규칙 기반 모델**과 **딥러닝 모델** 두 가지를 제공하며, 사용자는 FastAPI로 구축된 API 서버와 Streamlit으로 제작된 웹 인터페이스를 통해 쉽고 편리하게 자신의 에임을 분석할 수 있습니다.

## 🛠️ 기술 스택

- **AI / Machine Learning:**
  - **Object Detection:** YOLOv8
  - **Time-series Analysis:** LSTM (PyTorch)
  - **OCR (발사 감지):** EasyOCR
  - **ML-Ops:** Scikit-learn, Numpy, Pandas
- **Backend:**
  - **API Server:** FastAPI
  - **Web Server:** Uvicorn
- **Frontend:**
  - Streamlit
- **Core:**
  - Python 3.9+
  - OpenCV


## 🚀 설치 및 실행 방법

### 1. 사전 준비

- **Python 3.9 이상** 버전이 설치되어 있어야 합니다.
- **Git**이 설치되어 있어야 합니다.
- **NVIDIA GPU 및 CUDA:** 딥러닝 모델의 원활한 학습과 추론을 위해 NVIDIA GPU와 CUDA 환경 설정이 강력히 권장됩니다. CPU만으로도 실행은 가능하지만 속도가 매우 느립니다.

### 2. 프로젝트 클론 및 라이브러리 설치

```bash
# 1. 프로젝트를 다운로드합니다.
git clone https://github.com/baesiru/apexaimcoach.git
cd apexaimcoach

# 2. 필요한 모든 라이브러리를 설치합니다.
pip install -r requirements.txt
```

### 3. 데이터셋 준비 및 YOLO 모델 학습

**가장 먼저, 적군을 탐지하는 YOLOv8 커스텀 모델을 직접 학습시켜야 합니다.**

1.  **데이터셋 다운로드:**
    *   아래 링크의 Roboflow 데이터셋 페이지에 접속합니다.
    *   [**Apex Legends Characters Dataset (링크)**](https://universe.roboflow.com/online-resource-2/v1-jx7pl)
    *   원하는 버전을 선택한 후, `Export Dataset` -> `Format: YOLOv8` -> `download zip to computer`를 선택하여 ZIP 파일을 다운로드합니다.
    *   다운로드한 ZIP 파일의 압축을 풀어, 그 안의 모든 폴더와 파일을 프로젝트의 루트 디렉토리 아래 `v1.v2i.yolov8/` 폴더로 옮깁니다.
    *   **최종 폴더 구조:**
        ```
        /apex-aim-coach/
        ├── v1.v2i.yolov8/
        │   ├── train/
        │   ├── valid/
        │   ├── test/
        │   └── data.yaml
        └── ...
        ```

2.  **YOLO 모델 학습 실행:**
    *   프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 YOLO 모델 학습을 시작합니다.
    ```bash
    python train.py
    ```
    *   학습이 완료되면, `runs/detect/train/weights/best.pt` 경로에 학습된 모델 가중치 파일이 생성됩니다. 이 파일이 분석에 사용됩니다.

### 4. LSTM 모델 학습

**다음으로, 에임 패턴을 분석하는 LSTM 딥러닝 모델을 학습시킵니다.**

1.  **학습용 영상 준비:**
    *   본인의 플레이 영상을 **라벨링**해야 합니다.
    *   `low`, `good`, `high` 세 가지 감도 상태를 대표하는 영상을 준비합니다. 훈련장 영상과 실제 게임 영상을 섞는 것이 좋습니다.
    *   영상 파일들을 프로젝트 루트 디렉토리 또는 특정 폴더에 위치시킵니다.

2.  **피처 데이터 추출 (`.npy` 파일 생성):**
    *   `feature_extractor.py`를 실행하여 각 영상에서 시퀀스 데이터를 추출하고 `.npy` 파일로 저장합니다.
    *   아래는 예시입니다. (별도의 `run_extraction.py` 파일을 만들어 실행하면 편리합니다.)
        ```python
        # run_extraction.py (예시)
        from feature_extractor import extract_features_from_video

        YOLO_PATH = 'runs/detect/train/weights/best.pt'
        VIDEOS_TO_EXTRACT = {
            'C:/videos/my_low_sens_play.mp4': 30,
            'C:/videos/my_good_sens_play.mp4': 30,
            'C:/videos/my_high_sens_play.mp4': 30,
        }
        for video, seq_len in VIDEOS_TO_EXTRACT.items():
            extract_features_from_video(video, YOLO_PATH, seq_len)
        ```
    *   위 스크립트를 실행하면 `training_data` 폴더에 `my_low_sens_play_seq30_features.npy` 와 같은 파일들이 생성됩니다.

3.  **LSTM 모델 학습 실행:**
    *   `train_lstm.py` 파일을 열어 `LABELED_VIDEOS` 딕셔너리를 **실제 생성된 `.npy` 파일**을 사용하도록 수정하거나, 파일 이름 규칙에 맞게 파일들을 배치합니다. (현재 `train_lstm.py`는 파일 이름의 키워드를 보고 라벨을 자동 할당합니다.)
    *   터미널에서 아래 명령어를 실행합니다.
    ```bash
    python train_lstm.py
    ```
    *   학습이 완료되면 `lstm_sensitivity_analyzer.pth`와 `lstm_scaler.npy` 파일이 생성됩니다.

### 5. API 서버 및 프론트엔드 실행

**이제 모든 준비가 끝났습니다. 두 개의 터미널 창을 열어 각각 서버와 앱을 실행합니다.**

*   **터미널 1: FastAPI 백엔드 서버 실행**
  ```bash
  uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
  ```

*   **터미널 2: Streamlit 프론트엔드 앱 실행**
  ```bash
  streamlit run frontend.py
  ```

이제 자동으로 열리는 웹 브라우저(`http://localhost:8501`)를 통해 AI 에임 코치를 사용할 수 있습니다!

## ⚖️ 라이선스 (License)

### 프로젝트 코드

본 프로젝트의 소스 코드는 [MIT License](LICENSE) 하에 배포됩니다.

### 데이터셋

본 프로젝트의 YOLO 모델 학습에는 아래의 데이터셋이 사용되었으며, 해당 데이터셋의 라이선스 조항을 준수합니다.

- **데이터셋 이름:** v1 Computer Vision Project
- **제공자/원작자:** online resource 2
- **출처 링크:** [https://universe.roboflow.com/online-resource-2/v1-jx7pl](https://universe.roboflow.com/online-resource-2/v1-jx7pl)
- **라이선스:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

This project utilizes a dataset licensed under CC BY 4.0. We extend our gratitude to the original creators for making their work publicly available.