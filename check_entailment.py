from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ✅ 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
model = AutoModelForSequenceClassification.from_pretrained("Huffon/klue-roberta-base-nli")

# ✅ 추론 함수
def check_entailment(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    labels = ["entailment", "neutral", "contradiction"]
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    result = labels[predicted_index]

    return result, [round(p, 4) for p in probs]
