"""Trỏ lessonvid*.khaothi.ai về edge CDN trong nước.

DNS của máy (1.1.1.1) không gửi vị trí cho CDN, nên CDNetworks trả edge Singapore.
Edge Singapore trả mỗi đoạn .ts chậm hơn edge Hà Nội/Sài Gòn khoảng 4–5 lần.
8.8.8.8 có gửi vị trí, nên trả edge Việt Nam. Chỉ tên miền video bị đổi, DNS của máy giữ nguyên.

Tắt mặc định. Bật bằng ASR_CDN_DNS=8.8.8.8.
Đo 21/10/2026 trên video cũ: edge VN đạt 9–54x realtime, edge Singapore đạt 43–101x.
Edge VN nhanh với từng đoạn lẻ, nhưng chậm khi tải nhiều vì chưa có cache video cũ.
"""
import os
import random
import re
import socket
import subprocess
import time

RESOLVER = os.environ.get("ASR_CDN_DNS", "8.8.8.8")
_orig = socket.getaddrinfo
_cache = {}  # host → (ips, hết hạn)


def _lookup(host):
    ips, exp = _cache.get(host, ([], 0))
    if time.time() < exp:
        return ips
    try:
        out = subprocess.run(["dig", "+short", "+time=2", "+tries=1", f"@{RESOLVER}", host],
                             capture_output=True, text=True, timeout=5).stdout
        ips = [x for x in out.split() if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", x)]
    except Exception:
        ips = []
    _cache[host] = (ips, time.time() + (300 if ips else 60))   # lỗi thì dùng DNS máy, 1 phút sau thử lại
    return ips


def getaddrinfo(host, port, *args, **kwargs):
    if isinstance(host, str) and host.startswith("lessonvid") and host.endswith(".khaothi.ai"):
        ips = _lookup(host)
        if ips:
            return _orig(random.choice(ips), port, *args, **kwargs)
    return _orig(host, port, *args, **kwargs)


def install():
    socket.getaddrinfo = getaddrinfo
    return _lookup("lessonvid.khaothi.ai")
