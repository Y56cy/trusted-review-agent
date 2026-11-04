import re
from pypdf import PdfReader


# 非编号标题关键词：单独成块，避免粘进正文段落
_HEADINGS = {'摘要', '关键词', '参考文献', 'Abstract', 'Keywords', 'References'}


def _is_paragraph_end(raw_line):
    """行尾是否为段落结束标志：句末标点 + 两个及以上空格"""
    return bool(re.search(r'[。.!?？!][ \t]{2,}$', raw_line))


def _is_heading(line):
    """章节标题：非编号标题，或较短的编号标题（如 "2 相关工作"、"3.3.2 双边切片"）。

    参考文献条目（≥110 字）和以数字开头的正文行（如 "3 × 10 −4"）都更长，
    不会被误判为标题。
    """
    if line in _HEADINGS:
        return True
    return bool(re.match(r'^\d+(\.\d+)*\s+\S', line)) and len(line) <= 25


def _starts_new_block(line):
    """新块开头：参考文献条目（数字 + 英文作者名）或图表说明段。

    仅匹配 "N Author" 形态，避免把 "3 × 10 −4" 这类以数字开头的正文行误切。
    """
    if re.match(r'^\d+\s+[A-Z]', line):
        return True
    if re.match(r'^[图表]\s*\d', line):
        return True
    return False


def _merge_lines(lines):
    """把属于同一段落的多行拼接成一行。

    PDF 在换行处会插入一个空格，对中文而言该空格是换行伪影，
    对英文/数字/标点而言则是真实分词空格。因此按上一行末尾字符
    是否为 ASCII 决定是否保留空格。
    """
    if not lines:
        return ''
    merged = lines[0]
    for line in lines[1:]:
        if merged[-1].isascii():
            merged += ' ' + line
        else:
            merged += line
    return merged


def clean_text(text):
    """把 PDF 提取的碎片文本合并成连贯段落"""
    paragraphs = []
    current = []

    for raw in text.split('\n'):
        # probe 保留行尾空格用于段落结束判断，仅去掉 \r
        probe = raw.rstrip('\r')
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(_merge_lines(current))
                current = []
            continue

        # 章节标题独立成段，不与正文粘连
        if _is_heading(line):
            if current:
                paragraphs.append(_merge_lines(current))
                current = []
            paragraphs.append(line)
            continue

        if current and _starts_new_block(line):
            paragraphs.append(_merge_lines(current))
            current = []

        current.append(line)

        if _is_paragraph_end(probe):
            paragraphs.append(_merge_lines(current))
            current = []

    if current:
        paragraphs.append(_merge_lines(current))

    # 归一化段落内部多余空格
    paragraphs = [re.sub(r'[ \t]+', ' ', p) for p in paragraphs]
    return '\n'.join(paragraphs)


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