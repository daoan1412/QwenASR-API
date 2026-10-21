"""Thử tải: N job normal (xoay vòng các video), sau vài giây chen 2 job high. Dùng: load_test.py BASE N"""
import json, sys, time, urllib.request
from pathlib import Path

base, n = sys.argv[1], int(sys.argv[2])
key = Path("/media/daoan/Data/Projects/qwen3-asr/.api_key").read_text().strip()
URLS = ["https://lessonvid.khaothi.ai/2026/8/1202663/master.m3u8",
        "https://lessonvid.khaothi.ai/2026/8/25343720/master.m3u8",
        "https://lessonvid.khaothi.ai/2026/8/25337370/master.m3u8"]


def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "asr-load/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


t0 = time.time()
jobs = {call("POST", "/v1/asr/jobs", {"audio_url": URLS[i % 3], "ref": f"n{i}"})["job_id"]: "normal" for i in range(n)}
time.sleep(5)
for i in range(2):
    jobs[call("POST", "/v1/asr/jobs", {"audio_url": URLS[0], "ref": f"h{i}", "priority": "high"})["job_id"]] = "high"
done, audio = {}, 0.0
while len(done) < len(jobs):
    time.sleep(2)
    for jid in set(jobs) - set(done):
        r = call("GET", f"/v1/asr/jobs/{jid}")
        if r["status"] in ("succeeded", "failed", "cancelled"):
            done[jid] = r
            audio += (r.get("result") or {}).get("transcribed_sec") or 0
            if jobs[jid] == "high":
                print(f"HIGH {r['ref']} xong sau {time.time()-t0-5:.1f}s kể từ lúc gửi, timing={r.get('timing')}", flush=True)
wall = time.time() - t0
st = {}
for r in done.values():
    st[r["status"]] = st.get(r["status"], 0) + 1
rtfs = sorted(r["timing"]["rtf"] for r in done.values() if r.get("timing", {}).get("rtf"))
print(f"{len(jobs)} job trong {wall:.0f}s; {st}; audio {audio/3600:.2f} h → {audio/wall:.0f}x realtime "
      f"= {audio/wall:.0f} h video/h; rtf trung vị {rtfs[len(rtfs)//2]}")
