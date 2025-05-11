from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 한국어 자연어 추론 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large")

def check_entailment(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # 소프트맥스 확률 계산
    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()  # [entailment, neutral, contradiction]

    # 라벨 설정
    labels = ["entailment", "neutral", "contradiction"]
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    result = labels[predicted_index]

    # 소수점 4자리로 정리해서 반환
    return result, [round(p, 4) for p in probs]
