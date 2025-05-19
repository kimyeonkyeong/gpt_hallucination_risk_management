# ✅ 추가: GPU 지원 + padding 적용 + eval 설정
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
model = AutoModelForSequenceClassification.from_pretrained("Huffon/klue-roberta-base-nli")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def check_entailment(premise, hypothesis):
    if not premise or not isinstance(premise, str):
        return {"label": "error", "probs": [0.0, 0.0, 0.0], "confidence": 0.0}

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    if outputs.logits.shape[1] != 3:
        print("❗ logits shape mismatch:", outputs.logits.shape)
        return {"label": "error", "probs": [0.0, 0.0, 0.0], "confidence": 0.0}

    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    labels = ["entailment", "neutral", "contradiction"]
    predicted_index = torch.argmax(outputs.logits, dim=1).item()

    return {
        "label": labels[predicted_index],
        "probs": [round(p, 4) for p in probs],
        "confidence": round(probs[predicted_index], 4)
    }

