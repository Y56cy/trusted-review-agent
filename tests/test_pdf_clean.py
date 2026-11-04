import re
from pypdf import PdfReader

reader = PdfReader("../data/papers/基于双边网格滤波的3D高斯辐射场无人机SLAM方法_高连科.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    lines = text.split('\n')
    merged = []
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) <= 2:
            buffer += line
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(line)
    if buffer:
        merged.append(buffer)
    return '\n'.join(merged)

cleaned = clean_text(full_text)

# 只保留长度 >= 30 的段落，过滤掉标题、作者、单位等碎片
paragraphs = [p.strip() for p in cleaned.split('\n') if len(p.strip()) >= 30]

print(f"=== 共 {len(paragraphs)} 个有效段落 ===")
for i, p in enumerate(paragraphs[:10]):
    print(f"\n[段落 {i+1}] {p[:150]}")