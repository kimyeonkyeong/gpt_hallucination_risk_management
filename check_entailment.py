# ✅ 라이브러리 불러오기
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ 모델 및 토크나이저 로드 (Huffon/klue-roberta-base-nli)
tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
model = AutoModelForSequenceClassification.from_pretrained("Huffon/klue-roberta-base-nli")

# ✅ 디바이스 설정 (GPU 우선 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ 추론 함수 정의
def check_entailment(premise, hypothesis):
    # 입력 검증
    if not premise or not isinstance(premise, str) or len(premise.strip()) < 5:
        return {"label": "error", "probs": [0.0, 0.0, 0.0], "confidence": 0.0}

    # 토큰화 및 입력 전처리
    encoded = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 모델 추론
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

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
