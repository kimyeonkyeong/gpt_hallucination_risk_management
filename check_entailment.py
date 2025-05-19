from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
model = AutoModelForSequenceClassification.from_pretrained("Huffon/klue-roberta-base-nli")

def check_entailment(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    labels = ["entailment", "neutral", "contradiction"]  # ✅ 확인 필수: 이 모델은 해당 순서
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    return {
        "label": labels[predicted_index],
        "probs": [round(p, 4) for p in probs],
        "confidence": round(probs[predicted_index], 4)
    }
