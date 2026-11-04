import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = r"D:\dachuang\trusted-review-agent\models\Fengshenbang--Erlangshen-RoBERTa-110M-NLI\snapshots\master"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = ["contradiction", "neutral", "entailment"]

def nli_predict(premise, hypothesis, tag=""):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = labels[torch.argmax(probs).item()]
    print(f"--- {tag} ---")
    for label, p in zip(labels, probs):
        print(f"  {label}: {p.item():.4f}")
    print(f"  预测: {pred}")
    print()
    return pred

ev6 = "针对低空场景光照变化导致的图像外观光度不一致问题, 现有研究主要从特征稳健性、光度建模与学习补偿三方面展开. 传统方法如ORB-SLAM2 [9] 和DSO [10]等通过光度归一化, 直方图均衡或梯度特征增强在一定程度上提升了弱光与局部亮度变化下的匹配稳定性, 但依赖经验参数, 难以适应剧烈外观变化. 深度学习方法如LightSLAM [11]等借助光照不变特征与光度补偿网络提升系统的鲁棒性, 但计算复杂、推理延迟高、实时性受限. 基于NeRF的场景表征方法在外观一致性建模方面表现突出, 但体渲染计算量大. 总体来看, 现有抗光照vSLAM方法仍存在特征层与渲染层割裂, 实时性不足及光照自适应性弱等问题."

ev63 = "图3展示了本文提出的方法与PhotoSLAM在EuRoC V2-01序列的定性对比结果. 对比图3中红框内的图像区域, 本文所提出的方法对图像整体与细节重建显著优于PhotoSLAM. 其次, 在光照变化剧烈的场景, 本文的方法能更准确地重建场景外观, 这与表1中的定量评估指标一致,进一步验证了本文方法在外观渲染与建模质量上的显著优势."

ev69 = "实时外观重建. 本文的主要研究成果与结论如下: (1) 双边网格光度优化模块有效解决光照鲁棒性难题: 该模块将图像投影至三维双边空间, 并结合逐像素光度映射与可学习的仿射颜色变换, 对低空飞行过程中的光照变化进行自适应校正, 从而有效缓解复杂光照条件对建图与优化过程的干扰."

s1 = "这篇论文提出了一种双边网格光度优化模块来解决光照变化问题。"
s2 = "该模块将图像投影至三维双边空间，并结合逐像素光度映射与可学习的仿射颜色变换，对低空飞行过程中的光照变化进行自适应校正，从而有效缓解复杂光照条件对建图与优化过程的干扰。"

nli_predict(ev69, s1, "S1 vs ev69（同段，概括句）")
nli_predict(ev69, s2, "S2 vs ev69（同段，细节句）")
nli_predict(ev6,  s2, "S2 vs ev6（负样本对照）")
nli_predict(ev63, s2, "S2 vs ev63（负样本对照）")