import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from feature_extractor import extract_features_from_video
from lstm_model import create_lstm_model

LABELED_VIDEOS = {
    'slow1.mp4': 0,
    'slow2.mp4': 0,
    'slow3.mp4': 0,
    'good1.mp4': 1,
    'good2.mp4': 1,
    'good3.mp4': 1,
    'fast1.mp4': 2,
    'fast2.mp4': 2,
    'fast3.mp4': 2,
}
YOLO_MODEL_PATH = 'runs/detect/train/weights/best.pt'
NUM_FEATURES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    all_sequences, all_labels = [], []
    for video_file, label in LABELED_VIDEOS.items():
        if not os.path.exists(video_file):
            print(f"경고: '{video_file}'을 찾을 수 없습니다. 건너뜁니다.")
            continue
        sequences = extract_features_from_video(video_file, YOLO_MODEL_PATH)
        if sequences.shape[0] > 0:
            all_sequences.append(sequences)
            all_labels.extend([label] * sequences.shape[0])

    if not all_sequences:
        print("학습 데이터가 없습니다. LABELED_VIDEOS를 확인해주세요.");
        return

    X = np.vstack(all_sequences)
    y = np.array(all_labels)

    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))

    epsilon = 1e-8  # 0으로 나누는 것을 방지하기 위한 매우 작은 값
    X_normalized = (X - mean) / (std + epsilon)

    np.save('lstm_scaler.npy', {'mean': mean, 'std': std})

    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_lstm_model(input_size=NUM_FEATURES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- PyTorch 모델 학습 시작 (Device: {DEVICE}) ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            correct, total, val_loss = 0, 0, 0
            for sequences, labels in val_loader:
                sequences = sequences.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(
                f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), 'lstm_sensitivity_analyzer.pth')
    print("학습된 PyTorch 모델이 'lstm_sensitivity_analyzer.pth'로 저장되었습니다.")


if __name__ == '__main__':
    train()