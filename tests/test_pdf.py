from pypdf import PdfReader

reader = PdfReader("../data/papers/基于双边网格滤波的3D高斯辐射场无人机SLAM方法_高连科.pdf")
print(f"总页数：{len(reader.pages)}")

# 提取第 1 页文本
text = reader.pages[0].extract_text()
print("=== 第 1 页文本（前 500 字） ===")
print(text[:500])