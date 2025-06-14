import torch
import numpy as np
import os
from lstm_model import create_lstm_model
from feature_extractor import extract_features_from_video

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_recommendation_based_on_confidence(user_dpi, user_sens, predicted_class, confidence):
    current_edpi = user_dpi * user_sens
    adjustment_factor = 1.0
    class_labels = {0: "낮음", 1: "적절", 2: "높음"}

    analysis_text = f"LSTM 모델 분석 결과, 현재 감도는 '{class_labels[predicted_class]}'으로 판단됩니다. (신뢰도: {confidence * 100:.2f}%)"
    recommendation_detail = ""

    if predicted_class == 0:  # '낮음'으로 판단
        if confidence > 0.90:
            adjustment_factor = 1.15  # 15% 증가
            recommendation_detail = "매우 확실하게 감도가 낮습니다. 감도를 크게 높여보는 것을 권장합니다."
        elif confidence > 0.70:
            adjustment_factor = 1.08  # 8% 증가
            recommendation_detail = "감도가 낮은 경향이 있습니다. 감도를 약간 높여보세요."
        else:
            adjustment_factor = 1.04  # 4% 증가
            recommendation_detail = "감도가 미세하게 낮을 수 있습니다. 조금만 높여보는 것을 고려해보세요."

    elif predicted_class == 2:  # '높음'으로 판단
        if confidence > 0.90:
            adjustment_factor = 0.85  # 15% 감소
            recommendation_detail = "매우 확실하게 감도가 높습니다. 감도를 크게 낮추는 것을 권장합니다."
        elif confidence > 0.70:
            adjustment_factor = 0.92  # 8% 감소
            recommendation_detail = "감도가 높은 경향이 있습니다. 감도를 약간 낮춰보세요."
        else:
            adjustment_factor = 0.96  # 4% 감소
            recommendation_detail = "감도가 미세하게 높을 수 있습니다. 조금만 낮춰보는 것을 고려해보세요."

    else:  # '적절'로 판단 (predicted_class == 1)
        recommendation_detail = "현재 감도는 매우 안정적인 것으로 분석됩니다. 훌륭한 에임입니다!"

    new_edpi = current_edpi * adjustment_factor
    new_sens = new_edpi / user_dpi

    return {
        "summary": {
            "predicted_class_str": class_labels[predicted_class],
            "confidence": confidence
        },
        "recommendation": {
            "detail_text": recommendation_detail,
            "adjustment_factor": adjustment_factor
        },
        "current_settings": {
            "dpi": user_dpi,
            "sens": user_sens,
            "edpi": current_edpi
        },
        "recommended_settings": {
            "sens": new_sens,
            "edpi": new_edpi
        }
    }


def predict_sensitivity_with_lstm(video_path, yolo_model_path, lstm_model_path, user_dpi, user_sens):
    scaler_path = 'lstm_scaler.npy'
    if not all([os.path.exists(p) for p in [video_path, yolo_model_path, lstm_model_path, scaler_path]]):
        print("필요 파일(영상, 모델, 스케일러)이 없습니다. 경로를 확인해주세요.")
        return

    model = create_lstm_model().to(DEVICE)
    model.load_state_dict(torch.load(lstm_model_path, map_location=DEVICE))
    model.eval()

    scaler = np.load(scaler_path, allow_pickle=True).item()
    mean, std = scaler['mean'], scaler['std']

    sequences = extract_features_from_video(video_path, yolo_model_path)
    if sequences.shape[0] == 0:
        return "분석할 교전 데이터가 없습니다. 영상에 발사 장면이 있는지 확인해주세요."


    epsilon = 1e-8
    sequences_normalized = (sequences - mean) / (std + epsilon)
    sequences_tensor = torch.tensor(sequences_normalized, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(sequences_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    avg_probs = torch.mean(probabilities, dim=0)
    predicted_class = torch.argmax(avg_probs).item()
    confidence = avg_probs[predicted_class].item()

    recommendation = get_recommendation_based_on_confidence(
        user_dpi=user_dpi,
        user_sens=user_sens,
        predicted_class=predicted_class,
        confidence=confidence
    )

    return recommendation