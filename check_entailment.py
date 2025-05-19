from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-klue-nli")
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-klue-nli")

def check_entailment(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    labels = ["contradiction", "neutral", "entailment"]
    predicted_index = torch.argmax(outputs.logits, dim=1).item()
    result = labels[predicted_index]
    confidence = round(probs[predicted_index], 4)

    return {
        "label": result,
        "probs": [round(p, 4) for p in probs],
        "confidence": confidence
    }
