import torch
import torch.cuda as cuda
from ultralytics import YOLO

def train_model():
    if cuda.is_available():
        device = torch.device("cuda")
        print("CUDA 가능")
    else:
        device = torch.device("cpu")
        print("CUDA 불가능")

    model = YOLO('yolov8n.pt')
    data_path = 'v1.v2i.yolov8/data.yaml'
    device = 'cuda' if cuda.is_available() else 'cpu'

    print("모델 훈련을 시작합니다...")
    model.train(data=data_path, epochs=25, imgsz=800, device=device)

if __name__ == '__main__':
    train_model()
