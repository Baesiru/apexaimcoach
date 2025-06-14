import os
from recognition_rule_based import run_rule_based_analysis
from lstm_predictor import predict_sensitivity_with_lstm
import torch

def check_gpu_environment():
    print("\n" + "="*20 + " GPU 환경 진단 " + "="*20)
    if torch.cuda.is_available():
        print("✅ GPU 사용 가능 (CUDA is available)")
        print(f"   - 사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        print(f"   - 현재 GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"   - PyTorch 버전: {torch.__version__}")
        print(f"   - CUDA 버전 (PyTorch 빌드 시): {torch.version.cuda}")
    else:
        print("❌ GPU 사용 불가능 (CUDA is not available)")
        print("   - 현재 CPU로만 동작합니다. 분석 속도가 매우 느릴 수 있습니다.")
        print(f"   - PyTorch 버전: {torch.__version__}")
        print("   - PyTorch가 CPU 버전으로 설치되었거나, NVIDIA 드라이버/CUDA 설정에 문제가 있을 수 있습니다.")
    print("="*54 + "\n")
def start_recognition_process():
    print("\n--- 에임 분석을 시작합니다 ---")
    while True:
        video_path = input("분석할 동영상 파일의 경로를 입력하세요: ").strip().replace('"', '')
        if os.path.exists(video_path):
            break
        else:
            print(f"오류: 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요: '{video_path}'")

    while True:
        try:
            user_dpi = int(input("사용자의 마우스 DPI를 입력하세요 (예: 800): ").strip())
            if user_dpi > 0:
                break
            else:
                print("오류: DPI는 0보다 큰 정수여야 합니다.")
        except ValueError:
            print("오류: 유효한 숫자를 입력해주세요.")

    while True:
        try:
            user_sens = float(input("사용자의 인게임 감도를 입력하세요 (예: 1.25): ").strip())
            if user_sens > 0:
                break
            else:
                print("오류: 감도는 0보다 큰 숫자여야 합니다.")
        except ValueError:
            print("오류: 유효한 숫자를 입력해주세요.")

    while True:
        method = input("분석 방법을 선택하세요 (1: 규칙 기반, 2: LSTM 딥러닝): ").strip()
        if method in ['1', '2']:
            break
        else:
            print("오류: 1 또는 2를 입력해주세요.")

    print("\n입력된 정보로 분석을 시작합니다...")
    yolo_model_path = 'runs/detect/train/weights/best.pt'
    if not os.path.exists(yolo_model_path):
        print(f"오류: 훈련된 YOLO 모델 파일을 찾을 수 없습니다: {yolo_model_path}")
        return

    if method == '1':
        print("--- 규칙 기반 분석을 실행합니다 ---")
        run_rule_based_analysis(
            model_path=yolo_model_path,
            source_video_path=video_path,
            user_dpi=user_dpi,
            user_sens=user_sens
        )
    elif method == '2':
        print("--- PyTorch LSTM 딥러닝 분석을 실행합니다 ---")
        lstm_model_path = 'lstm_sensitivity_analyzer.pth'
        if not os.path.exists(lstm_model_path):
            print(f"오류: 학습된 PyTorch LSTM 모델('{lstm_model_path}')을 찾을 수 없습니다.")
            print("먼저 `train_lstm.py`를 실행하여 모델을 학습시켜주세요.")
            return

        recommendation = predict_sensitivity_with_lstm(
            video_path=video_path,
            yolo_model_path=yolo_model_path,
            lstm_model_path=lstm_model_path,
            user_dpi=user_dpi,
            user_sens=user_sens
        )
        print("\n[ PyTorch LSTM 감도 분석 결과 ]\n")
        print(recommendation)
        print("\n" + "=" * 56)


def interactive_main():
    print("=" * 40)
    print("   Apex 레전드 에임 분석 프로그램   ")
    print("=" * 40)
    while True:
        command = input("실행할 명령어를 입력하세요 (analyze / exit): ").strip().lower()
        if command == 'analyze':
            start_recognition_process()
            break
        elif command == 'exit':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 명령어입니다. 'analyze' 또는 'exit'을 입력해주세요.")

if __name__ == '__main__':
    interactive_main()