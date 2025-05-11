from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large")

def check_entailment(premise, hypothesis):
    # 입력 문장 토크나이즈
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # softmax 확률 계산
    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()

    # 결과 라벨 지정
    labels = ["entailment", "neutral", "contradiction"]
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    result = labels[predicted_index]

    return result, [round(p, 4) for p in probs]
