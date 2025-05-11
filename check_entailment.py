from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large")

def check_entailment(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    labels = ["entailment", "neutral", "contradiction"]
    result = labels[torch.argmax(probs)]
    return result, probs.tolist()[0]