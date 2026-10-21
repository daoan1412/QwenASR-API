"""Gửi job HLS lên server, chờ xong, in tóm tắt và lưu kết quả. Dùng: try_jobs.py BASE OUT_DIR [ref=url[@max_seconds] ...]"""
import json, sys, time, urllib.request
from pathlib import Path

base, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
key = Path("/media/daoan/Data/Projects/qwen3-asr/.api_key").read_text().strip()


def call(method, path, body=None, headers=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "asr-try/1.0",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read())


jobs, t0 = {}, time.time()
for arg in sys.argv[3:]:
    ref, url = arg.split("=", 1)
    url, _, mx = url.partition("@")
    code, r = call("POST", "/v1/asr/jobs", {"audio_url": url, "ref": ref, "max_seconds": float(mx or 0)},
                   {"Idempotency-Key": f"{ref}-{url}-{time.time()}"})
    jobs[r["job_id"]] = ref
    print(code, r["job_id"], ref, r["status"], r.get("queue_position"), flush=True)

pending = set(jobs)
while pending:
    time.sleep(2)
    for jid in list(pending):
        _, r = call("GET", f"/v1/asr/jobs/{jid}")
        if r["status"] in ("succeeded", "failed", "cancelled"):
            pending.discard(jid)
            (out_dir / f"{jobs[jid]}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
            res, tm = r.get("result") or {}, r.get("timing") or {}
            print(f"{time.time()-t0:6.1f}s {jobs[jid]:>9} {r['status']} {res.get('status')} "
                  f"dur={res.get('duration_sec')} tx={res.get('transcribed_sec')} cues={len(res.get('cues') or [])} "
                  f"lang={res.get('language')} timing={tm} err={r.get('error')}", flush=True)
print(f"tổng {time.time()-t0:.1f}s")
