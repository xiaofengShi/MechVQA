from huggingface_hub import snapshot_download
import requests
import os
import httpx
import huggingface_hub

# 设置全局请求超时
requests.adapters.DEFAULT_TIMEOUT = 120

# 禁用缓存
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# 设置 httpx 超时
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

huggingface_hub.constants.DEFAULT_TIMEOUT = httpx.Timeout(
    timeout=120.0,  # 总超时时间
    connect=60.0,  # 连接超时
    read=120.0,  # 读取超时
    write=60.0  # 写入超时
)

# snapshot_download(
#     repo_id="RZ-Q/mechrlv2",
#     repo_type="dataset",
#     local_dir="./data",
#     allow_patterns=["images/*"],
#     max_workers=4,  # 降低并发数，避免连接过多
#     resume_download=True,
#     cache_dir=None,
#     local_dir_use_symlinks=False)


snapshot_download(
    repo_id="RZ-Q/MDU-VQA",
    repo_type="dataset",
    local_dir="./data/images",
    allow_patterns=["train/*"],
    max_workers=4,  # 降低并发数，避免连接过多
    resume_download=True,
    cache_dir=None,
    local_dir_use_symlinks=False)
