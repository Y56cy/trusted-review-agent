from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = r"D:\dachuang\trusted-review-agent\models\Fengshenbang--Erlangshen-RoBERTa-110M-NLI\snapshots\master"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

def nli_predict(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True,max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    labels = ["contradiction", "neutral", "entailment"]
    for label, p in zip(labels, probs):
        print(f"{label}: {p.item():.4f}")
    return labels[torch.argmax(probs).item()]

print("--- 测试1 ---")
nli_predict("今天下雨了", "地面是湿的")
print("--- 测试2 ---")
nli_predict("今天下雨了", "今天是晴天")
print("--- 测试3 ---")
nli_predict("今天下雨了", "我喜欢吃苹果")