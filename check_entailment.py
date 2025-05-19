# ✅ 라이브러리 불러오기
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-nli")
model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-nli")
model.eval()

# ✅ CPU 전용 추론 함수
def check_entailment(premise, hypothesis):
    # 입력 검증
    if not premise or not isinstance(premise, str) or len(premise.strip()) < 5:
        return {"label": "error", "probs": [0.0, 0.0, 0.0], "confidence": 0.0}

    # 토큰화 및 입력 전처리 (device 전송 없이 CPU 사용)
    encoded = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 모델 추론 (CPU 기준)
    with torch.no_grad():
        outputs = model(**encoded)

    # 확률 계산
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze().tolist()

    # 라벨 정의
    labels = ["entailment", "neutral", "contradiction"]
    predicted_index = torch.argmax(logits, dim=1).item()

    # 결과 반환
    return {
        "label": labels[predicted_index],
        "probs": [round(p, 4) for p in probs],
        "confidence": round(probs[predicted_index], 4)
    }
