import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import time
import easyocr

try:
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("규칙 기반 모듈용 EasyOCR Reader가 초기화되었습니다.")
    if torch.cuda.is_available():
        print("EasyOCR이 GPU(CUDA)를 사용합니다.")
    else:
        print("EasyOCR이 CPU를 사용합니다.")
except Exception as e:
    print(f"EasyOCR 초기화 중 오류 발생: {e}")
    reader = None


def get_dynamic_ammo_roi(frame_width, frame_height):
    BASE_WIDTH, BASE_HEIGHT = 1920, 1080
    BASE_ROI = (1725, 960, 60, 40)
    x_ratio, y_ratio = BASE_ROI[0] / BASE_WIDTH, BASE_ROI[1] / BASE_HEIGHT
    w_ratio, h_ratio = BASE_ROI[2] / BASE_WIDTH, BASE_ROI[3] / BASE_HEIGHT
    return (int(frame_width * x_ratio), int(frame_height * y_ratio),
            int(frame_width * w_ratio), int(frame_height * h_ratio))


def get_sensitivity_recommendation(current_dpi, current_sens, total_seconds, overshoot_count, undershoot_frames,
                                   video_fps=60):
    current_edpi = current_dpi * current_sens
    if total_seconds < 3:
        return {"summary": {"predicted_class_str": "분석 불가"},
                "recommendation": {"detail_text": "분석된 교전 데이터가 충분하지 않아(3초 미만) 추천할 수 없습니다."},
                "current_settings": {"dpi": current_dpi, "sens": current_sens, "edpi": current_edpi},
                "recommended_settings": {}, "rule_based_stats": {"total_shooting_seconds": total_seconds}}
    overshoots_per_sec = overshoot_count / total_seconds if total_seconds > 0 else 0
    undershoot_rate = undershoot_frames / (total_seconds * video_fps) if total_seconds > 0 else 0
    recommendation_text, adjustment_factor, predicted_class = "", 1.0, "적절"
    if overshoots_per_sec > 0.4:
        adjustment_factor, recommendation_text, predicted_class = 0.88, "오버슈팅이 매우 잦습니다. 현재 감도가 너무 높은 것으로 보입니다.", "높음"
    elif overshoots_per_sec > 0.2:
        adjustment_factor, recommendation_text, predicted_class = 0.94, "오버슈팅 경향이 있습니다. 감도를 약간 낮춰보는 것을 추천합니다.", "높음"
    elif undershoot_rate > 0.15:
        adjustment_factor, recommendation_text, predicted_class = 1.10, "트래킹 시 에임이 적을 따라가지 못하는 경향이 강합니다. 감도가 낮을 수 있습니다.", "낮음"
    elif undershoot_rate > 0.07:
        adjustment_factor, recommendation_text, predicted_class = 1.05, "트래킹이 약간 느립니다. 감도를 약간 높여보는 것을 추천합니다.", "낮음"
    else:
        recommendation_text = f"현재 감도(eDPI: {current_edpi:.2f})는 안정적인 범위에 있습니다. 훌륭한 에임입니다!"
    new_edpi, new_sens = current_edpi * adjustment_factor, (current_edpi * adjustment_factor) / current_dpi
    return {"summary": {"predicted_class_str": predicted_class, "confidence": 1.0},
            "recommendation": {"detail_text": recommendation_text, "adjustment_factor": adjustment_factor},
            "current_settings": {"dpi": current_dpi, "sens": current_sens, "edpi": current_edpi},
            "recommended_settings": {"sens": new_sens, "edpi": new_edpi},
            "rule_based_stats": {"overshoot_count": overshoot_count, "undershoot_frames": undershoot_frames,
                                 "total_shooting_seconds": total_seconds}}


def run_rule_based_analysis(model_path, source_video_path, user_dpi, user_sens):
    if reader is None: return {"error": "EasyOCR 리더가 초기화되지 않았습니다."}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        print(f"비디오 열기 실패: {source_video_path}")
        return {"error": f"비디오 열기 실패: {source_video_path}"}

    original_width, original_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 60

    AMMO_ROI = get_dynamic_ammo_roi(original_width, original_height)
    CROSSHAIR_POS = (original_width // 2, original_height // 2)

    prev_ammo_count, shooting_state_persistence, SHOOTING_PERSISTENCE_FRAMES = -1, 0, 15
    overshoot_count, undershoot_frames, shooting_frames = 0, 0, 0
    prev_target_vector_x, prev_target_pos = 0, None

    start_time = time.time()
    total_frames_processed = 0
    print("규칙 기반 분석을 시작합니다 (최적화 버전)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        total_frames_processed += 1

        is_shooting_this_frame = False
        closest_enemy_info = None

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
        if is_shooting:
            shooting_state_persistence -= 1
            shooting_frames += 1

            results = model.predict(frame, conf=0.5, verbose=False, device=device)
            min_distance = float('inf')
            for box in results[0].boxes.xyxy.cpu():
                enemy_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                distance = np.linalg.norm(np.array(CROSSHAIR_POS) - np.array(enemy_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_enemy_info = {"center": enemy_center, "distance": distance}

            if closest_enemy_info:
                target_pos = closest_enemy_info["center"]
                current_target_vector_x = target_pos[0] - CROSSHAIR_POS[0]
                if np.sign(current_target_vector_x) != np.sign(
                        prev_target_vector_x) and prev_target_vector_x != 0 and abs(current_target_vector_x) > 5:
                    overshoot_count += 1
                if prev_target_pos is not None:
                    target_speed = np.linalg.norm(np.array(target_pos) - np.array(prev_target_pos))
                    if target_speed > 1.5 and closest_enemy_info["distance"] > 25:
                        undershoot_frames += 1
                prev_target_vector_x = current_target_vector_x

        if closest_enemy_info:
            prev_target_pos = closest_enemy_info["center"]
        else:
            prev_target_pos = None
            prev_target_vector_x = 0

    processing_time = time.time() - start_time
    print(f"분석 완료. (총 {total_frames_processed} 프레임, 소요 시간: {processing_time:.2f}초)")
    cap.release()
    total_shooting_seconds = shooting_frames / video_fps

    recommendation_dict = get_sensitivity_recommendation(
        user_dpi, user_sens, total_shooting_seconds,
        overshoot_count, undershoot_frames, video_fps
    )
    return recommendation_dict