from modelscope import snapshot_download

model_dir = snapshot_download(
    'BAAI/bge-small-zh-v1.5',
    local_dir='../models/bge-small-zh-v1.5'
)
print("下载完成，模型路径：", model_dir)