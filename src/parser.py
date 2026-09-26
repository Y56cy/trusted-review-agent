import re
from pypdf import PdfReader


def clean_text(text):
    """把 PDF 提取的碎片文本合并成连贯段落"""
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


def parse_pdf(pdf_path, min_length=30):
    """
    解析 PDF，返回段落列表。
    每个段落是一个字典：{"text": ..., "page": ..., "para_id": ...}
    """
    reader = PdfReader(pdf_path)
    all_paragraphs = []
    para_counter = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() + "\n"
        cleaned = clean_text(page_text)
        paras = [p.strip() for p in cleaned.split('\n') if len(p.strip()) >= min_length]
        for p in paras:
            para_counter += 1
            all_paragraphs.append({
                "text": p,
                "page": page_num,
                "para_id": para_counter
            })

    return all_paragraphs