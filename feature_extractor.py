import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import torch

try:
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("피처 추출기용 EasyOCR Reader가 초기화되었습니다.")
except Exception as e:
    print(f"EasyOCR 초기화 중 오류 발생: {e}")
    reader = None

def extract_sequences_from_burst(burst_data, sequence_length):
    num_sequences = len(burst_data) - sequence_length + 1
    if num_sequences <= 0:
        return []

    sequences = []
    for i in range(num_sequences):
        sequence = burst_data[i:i + sequence_length]
        sequences.append(sequence)
    return sequences


def get_dynamic_ammo_roi(frame_width, frame_height):
    BASE_WIDTH, BASE_HEIGHT = 1920, 1080
    BASE_ROI = (1725, 960, 60, 40)

    x_ratio = BASE_ROI[0] / BASE_WIDTH
    y_ratio = BASE_ROI[1] / BASE_HEIGHT
    w_ratio = BASE_ROI[2] / BASE_WIDTH
    h_ratio = BASE_ROI[3] / BASE_HEIGHT

    new_x = int(frame_width * x_ratio)
    new_y = int(frame_height * y_ratio)
    new_w = int(frame_width * w_ratio)
    new_h = int(frame_height * h_ratio)

    return (new_x, new_y, new_w, new_h)


def extract_features_from_video(video_path, model_path, sequence_length=30, debug_visualization=False):
    if reader is None:
        print("EasyOCR 리더가 초기화되지 않아 피처 추출을 중단합니다.")
        return np.array([])

    print(f"'{os.path.basename(video_path)}'에서 피처 추출을 시작합니다 (최적화 버전)...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 열기 실패: {video_path}")
        return np.array([])

    original_width, original_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    AMMO_ROI = get_dynamic_ammo_roi(original_width, original_height)
    CROSSHAIR_POS = np.array([original_width // 2, original_height // 2])
    NUM_FEATURES = 6

    all_feature_sequences, current_shooting_burst = [], []
    prev_ammo_count, shooting_state_persistence = -1, 0
    SHOOTING_PERSISTENCE_FRAMES = 15
    prev_target_pos = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        closest_enemy_pos = None
        is_shooting_this_frame = False
        try:
            x, y, w, h = AMMO_ROI
            roi = frame[y:y + h, x:x + w]
            ocr_results = reader.readtext(roi, detail=0, allowlist='0123456789')
            ocr_text = ''.join(ocr_results)
            if ocr_text.isdigit():
                current_ammo = int(ocr_text)
                if 0 <= current_ammo < prev_ammo_count: is_shooting_this_frame = True
                if current_ammo >= 0: prev_ammo_count = current_ammo
        except Exception:
            pass
        if is_shooting_this_frame: shooting_state_persistence = SHOOTING_PERSISTENCE_FRAMES
        is_shooting = shooting_state_persistence > 0
        if is_shooting: shooting_state_persistence -= 1

        if is_shooting:
            results = model.predict(frame, conf=0.4, verbose=False)
            closest_enemy_pos = None
            min_dist = float('inf')
            for box in results[0].boxes.xyxy.cpu():
                enemy_center = np.array([int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)])
                dist = np.linalg.norm(CROSSHAIR_POS - enemy_center)
                if dist < min_dist: min_dist, closest_enemy_pos = dist, enemy_center

            if closest_enemy_pos is not None:
                dist_vector = closest_enemy_pos - CROSSHAIR_POS
                target_velocity = closest_enemy_pos - prev_target_pos if prev_target_pos is not None else np.array(
                    [0, 0])
                features = np.concatenate([dist_vector, target_velocity, np.zeros(2)])
            else:
                features = np.zeros(NUM_FEATURES)
            current_shooting_burst.append(features)

        else:
            if len(current_shooting_burst) >= sequence_length:
                print(f"사격 이벤트 종료. {len(current_shooting_burst)} 프레임 길이의 burst에서 시퀀스 추출...")
                new_sequences = extract_sequences_from_burst(current_shooting_burst, sequence_length)
                if new_sequences:
                    all_feature_sequences.extend(new_sequences)
                    print(f"--> {len(new_sequences)}개의 시퀀스 추가. (총: {len(all_feature_sequences)}개)")
            current_shooting_burst = []

        prev_target_pos = closest_enemy_pos if closest_enemy_pos is not None else None

    if len(current_shooting_burst) >= sequence_length:
        new_sequences = extract_sequences_from_burst(current_shooting_burst, sequence_length)
        if new_sequences: all_feature_sequences.extend(new_sequences)

    cap.release()
    print(f"\n피처 추출 최종 완료. 총 {len(all_feature_sequences)}개의 시퀀스가 생성되었습니다.")
    return np.array(all_feature_sequences)