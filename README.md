# Qwen3-ASR job server

Server ASR dùng chung: Qwen3-ASR-1.7B (vLLM) + Qwen3-ForcedAligner-0.6B.
Chép lời video HLS (kết quả khớp `ml.VideoTranscript` bản v2 của moon-ml), tệp upload hoặc link tệp tiếng/video.
Từ 31/10/2026 mọi chỗ cần nhận dạng giọng chỉ gọi server này, không tự nạp mô hình:

| Ai gọi | Chế độ | Đầu vào |
|---|---|---|
| moon-ml `video_transcript` | `cues` | link HLS |
| ai-video-skills `speech_check.py` (Xưởng video, cả trong sandbox OpenShell) | `text`, `language: Vietnamese` | upload wav |
| ai-video-skills `align_words.py` (mốc từng chữ phụ đề) | `align` | upload wav + chữ TTS đọc |
| tiktok-phan-tich-video `chep_loi.py` | `cues` | upload mp4 |

Client stdlib dùng chung: `ai-video-skills/skills/video-core/kit/scripts/asr_client.py`.

Máy chính (192.168.4.63): user unit `~/.config/systemd/user/qwen3-asr-local.service`, nghe `127.0.0.1:8020` và
`172.17.0.1:8020` (docker0, cho sandbox), `ASR_GPU_UTIL=0.085` cạnh vLLM Qwen 27B. Bản ở máy khác (21–31/10/2026):
https://asr.daoan.online, unit `deploy/qwen3-asr.service`.

## Cài

```bash
uv venv -p 3.12 .venv
PIP_CONFIG_FILE=/dev/null uv pip install -p .venv/bin/python -r requirements.txt
python3 -c "import secrets;print('sk-asr-'+secrets.token_urlsafe(24))" > .api_key && chmod 600 .api_key
cp deploy/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now qwen3-asr moon-asr-tunnel
```

Tunnel cần token chạy tunnel ở `~/.cloudflared/moon-asr.token`.
Token này không nằm trong repo.

## Chạy

```bash
systemctl --user restart qwen3-asr        # server, 127.0.0.1:8000
systemctl --user restart moon-asr-tunnel  # Cloudflare tunnel → asr.daoan.online
journalctl --user -u qwen3-asr -f         # log
```

Server dùng khoảng 15–18 GB VRAM.
Strata không chạy cùng lúc được.

## API

Mọi endpoint trừ `/health` cần key.
Gửi key qua `Authorization: Bearer <key>` hoặc `X-API-Key: <key>`.
Key chính nằm trong `.api_key` (tên `main`, không giới hạn).
Khoá phụ ở `.api_keys`, mỗi dòng `tên khoá` (vd. `studio` cho Xưởng video); mỗi khoá phụ tối đa `ASR_KEY_MAX_QUEUE` job chờ.

Cloudflare chặn User-Agent `Python-urllib`.
Client dùng urllib phải tự đặt header `User-Agent`.

| Endpoint | Việc |
|---|---|
| `GET /health` | `{"status","queue","running","model","version"}` |
| `POST /v1/uploads` | Thân request là nguyên tệp (wav, mp3, mp4…) → `{upload_id, bytes}`; giữ 24 h từ lần dùng cuối |
| `POST /v1/asr/jobs` | Tạo job, trả `202` |
| `GET /v1/asr/jobs/{id}?wait=30` | Trạng thái và kết quả; `wait` giữ request tới khi xong (≤ 60 s) |
| `DELETE /v1/asr/jobs/{id}` | Huỷ job |
| `GET /v1/asr/jobs/{id}/result.vtt` | Phụ đề VTT |

```bash
curl https://asr.daoan.online/v1/asr/jobs \
  -H "Authorization: Bearer $(cat .api_key)" -H "Idempotency-Key: 1181195" \
  -H "Content-Type: application/json" \
  -d '{"audio_url":"https://lessonvid.khaothi.ai/2026/10/1181195/master.m3u8","ref":"1181195","max_seconds":7200}'
```

Job: đúng một trong `audio_url` / `upload_id`; `mode`:
- `cues` (mặc định): câu phụ đề `{s,e,t,lang}` + `vtt`, thuật toán v2 hai lượt, khoá ngôn ngữ Việt/Anh.
- `text`: chỉ `text`, khúc ~20 s nối lại. `language` (vd. `Vietnamese`) ép ngôn ngữ như `qwen_asr transcribe(language=…)`.
- `align`: cần `text`; trả `words` `[[từ, bắt đầu, kết thúc], …]` (tiếng ≤ 180 s). Tệp upload để qwen_asr tự đọc:
  mốc giống hệt `align_words.py` local cũ (đo 31/10/2026 trên 3 đoạn TTS).
  Hoặc `cues` `[{s, e, t, lang}]` thay `text` (10/11/2026): gán mốc từng từ cho lời ĐÃ CHÉP (vd. `ml.VideoTranscript`) mà
  không chép lại — tiếng dài bao nhiêu cũng được; mặc định mỗi câu một khúc trong khung giờ của câu (`options.align_chunk_s`
  0; khúc 150 s thì aligner trôi — 10% câu lệch > 26 s); `words` = `[[từ, s, e, câu], …]`, `câu` = chỉ số câu chứa từ
  (aligner có thể tách từ khác khoảng trắng). Video 53 phút ~37 s.
- `options.words: true` ở chế độ `cues`: chép xong thì gán mốc từng từ luôn, kết quả có thêm `words` `[[từ, s, e, câu], …]`
  (mốc từ đầu video) và `timing.align_sec`; aligner hỏng thì `words: null` + `words_error`, câu phụ đề vẫn trả.

`audio_url` đuôi `.m3u8`/`.ts` đi thẳng HLS như trước. Link khác: tải về, nội dung `#EXTM3U` → HLS, trang web →
`failed not_media`, còn lại giải mã bằng PyAV.

Hành vi chính:
- `priority`: `high`, `normal` hoặc `low`. Job `high` được thêm 4 slot chạy riêng.
- `Idempotency-Key` trùng thì trả job cũ. Nếu job cũ là `no_video`, `failed` hoặc `cancelled`, server tạo job mới.
- Playlist hoặc một đoạn `.ts` trả 403, 404, 410, hay HLS rỗng, thì job vẫn `succeeded`, với `result.status = "no_video"`.
  Server không thử lại các mã này.
- Lỗi mạng, OOM hay engine chết có `retryable: true`. Lỗi giải mã có `retryable: false`.
- Kết quả giữ 24 giờ trong `jobs/`. Job đang chờ được chạy lại khi server khởi động lại.
- Nếu có `callback_url`, server POST kết quả tới đó khi job kết thúc. Server thử lại tối đa 4 lần.

## Code

- `asrserver/hls.py`, `asr.py`, `cues.py`, `pipeline.py`: chép từ moon-ml v2.
  `hls.py` chỉ thêm phần tải trước, thử lại và báo tiến độ. Wav ra không đổi.
- `asrserver/langlock.py`: khoá token ngay sau `language`. Chỉ cho Vietnamese, English, None.
- `asrserver/engine.py`: gọi vLLM AsyncLLM. Mỗi khúc là một request, khúc của mọi job dùng chung batch.
- `asrserver/app.py`: API, hàng đợi, lưu job, upload, chế độ text/align, nhiều khoá.
- `asrserver/media.py`: tệp upload / link tệp tiếng-video → tiếng 16 kHz (PyAV).
- `asrserver/aligner.py`: Qwen3-ForcedAligner-0.6B trong tiến trình API (~1,5 GB VRAM), một yêu cầu một lúc.
- `tests/test_app.py`: test API với engine/aligner giả (`.venv/bin/python -m pytest -q tests`).
- `asrserver/edge_dns.py`: chọn edge CDN qua DNS khác. Tắt mặc định.
- `tools/try_jobs.py`, `tools/load_test.py`: gửi job thử và thử tải.

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ASR_HOST` | `127.0.0.1` | Một hoặc nhiều địa chỉ, cách nhau dấu phẩy (vd. `127.0.0.1,172.17.0.1`) |
| `ASR_GPU_UTIL` | `0.45` | Phần VRAM vLLM giữ. Encoder dùng thêm khoảng 3 GB ngoài phần này. |
| `ASR_ALIGN_DEVICE` | `cuda:0` | Thiết bị của ForcedAligner; `ASR_ALIGN_PRELOAD=0` thì nạp lúc job align đầu tiên |
| `ASR_MAX_UPLOAD_MB` | `512` | Cỡ tối đa một tệp upload / tệp tải từ link |
| `ASR_KEY_MAX_QUEUE` | `200` | Job chờ tối đa của mỗi khoá phụ |
| `ASR_MAX_ACTIVE` | `12` | Số job tải và chép cùng lúc. 6 thì mạng còn trống. |
| `ASR_HIGH_EXTRA` | `4` | Slot thêm cho job `high` |
| `ASR_MAX_QUEUE` | `2000` | Quá số này thì trả `429` |
| `ASR_LANG_LOCK` | `1` | `0` thì tắt khoá ngôn ngữ, version thành `v2-nolock` |
| `VLLM_BATCH_INVARIANT` | `1` | Kết quả ít phụ thuộc cách gom batch |
| `ASR_CDN_DNS` | trống | Đặt `8.8.8.8` để tải qua edge CDN Việt Nam. Tắt vì đo không nhanh hơn. |

## Số đo (21/10/2026, RTX 4090)

- 50 job (8.85 giờ audio) xong trong 96 s. Tức khoảng 330 giờ video mỗi giờ.
- Video 2 giờ chạy riêng: RTF 0.0103. Phần lớn thời gian là tải HLS (57 s), ASR mất 17.6 s.
- Mạng tải khoảng 8–11 MB/s là nút thắt. Luồng 480p tốn khoảng 280 MB cho mỗi giờ video.
- Hai lần chạy cùng video có thể khác nhau khoảng 0.4% số từ. Video ngắn thường ra giống hệt.
