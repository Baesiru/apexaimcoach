import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Annotated
from lstm_predictor import predict_sensitivity_with_lstm
from recognition_rule_based import run_rule_based_analysis

app = FastAPI(
    title="Apex Legends Aim Coach API",
    description="플레이 영상을 분석하여 최적의 감도를 추천하는 API입니다. LSTM 딥러닝 모델과 규칙 기반 모델을 모두 제공합니다.",
    version="1.1.0"
)

YOLO_MODEL_PATH = 'runs/detect/train/weights/best.pt'
LSTM_MODEL_PATH = 'lstm_sensitivity_analyzer.pth'
SCALER_PATH = 'lstm_scaler.npy'

if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLO 모델을 찾을 수 없습니다: {YOLO_MODEL_PATH}")


def save_temp_video(video_file: UploadFile):
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, video_file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    return temp_path


@app.get("/", summary="API 상태 확인")
def read_root():
    return {"message": "Apex Legends Aim Coach API is running!"}


@app.post("/analyze/lstm", summary="LSTM 모델로 감도 분석", tags=["Analysis"])
async def analyze_sensitivity_lstm(
        background_tasks: BackgroundTasks,
        video_file: Annotated[UploadFile, File(description="분석할 플레이 영상 파일")],
        user_dpi: Annotated[int, Form(description="사용자의 마우스 DPI (예: 800)")],
        user_sens: Annotated[float, Form(description="사용자의 인게임 감도 (예: 1.2)")]
):

    if not all([os.path.exists(p) for p in [LSTM_MODEL_PATH, SCALER_PATH]]):
        raise HTTPException(status_code=503, detail=f"LSTM 모델 또는 스케일러 파일이 준비되지 않았습니다. 'train_lstm.py'를 먼저 실행해주세요.")

    if not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"동영상 파일만 업로드할 수 있습니다. (업로드된 타입: {video_file.content_type})")

    temp_video_path = save_temp_video(video_file)
    background_tasks.add_task(os.remove, temp_video_path)

    try:
        print(f"LSTM 분석 시작: {video_file.filename}, DPI: {user_dpi}, Sens: {user_sens}")
        recommendation = predict_sensitivity_with_lstm(
            video_path=temp_video_path,
            yolo_model_path=YOLO_MODEL_PATH,
            lstm_model_path=LSTM_MODEL_PATH,
            user_dpi=user_dpi,
            user_sens=user_sens
        )
        print("LSTM 분석 완료.")
        return {"analysis_method": "LSTM", "result": recommendation}

    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 서버 오류 발생: {str(e)}")


@app.post("/analyze/rule-based", summary="규칙 기반으로 감도 분석", tags=["Analysis"])
async def analyze_sensitivity_rule_based(
        background_tasks: BackgroundTasks,
        video_file: Annotated[UploadFile, File(description="분석할 플레이 영상 파일")],
        user_dpi: Annotated[int, Form(description="사용자의 마우스 DPI")],
        user_sens: Annotated[float, Form(description="사용자의 인게임 감도")]
):
    if not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"동영상 파일만 업로드할 수 있습니다. (업로드된 타입: {video_file.content_type})")

    temp_video_path = save_temp_video(video_file)
    background_tasks.add_task(os.remove, temp_video_path)

    try:
        print(f"규칙 기반 분석 시작: {video_file.filename}, DPI: {user_dpi}, Sens: {user_sens}")

        recommendation_dict = run_rule_based_analysis(
            model_path=YOLO_MODEL_PATH,
            source_video_path=temp_video_path,
            user_dpi=user_dpi,
            user_sens=user_sens
        )

        print("규칙 기반 분석 완료.")

        return {"analysis_method": "Rule-based", "result": recommendation_dict}

    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 서버 오류 발생: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)