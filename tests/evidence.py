import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_pdf

pdf_path = r"D:\dachuang\trusted-review-agent\data\papers\基于双边网格滤波的3D高斯辐射场无人机SLAM方法_高连科.pdf"
paras = parse_pdf(pdf_path)

print(f"总段落数: {len(paras)}")
print()

target_ids = {6, 63, 69}
for p in paras:
    if p["para_id"] in target_ids:
        print(f"=== para_id={p['para_id']} | page={p['page']} ===")
        print(p["text"])
        print()