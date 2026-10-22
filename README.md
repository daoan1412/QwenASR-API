# Qwen3-ASR job server

Server chép lời video HLS bằng Qwen3-ASR-1.7B.
Kết quả khớp `ml.VideoTranscript` bản v2 của moon-ml.
Địa chỉ public: https://asr.daoan.online.

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
Key nằm trong `.api_key`.

Cloudflare chặn User-Agent `Python-urllib`.
Client dùng urllib phải tự đặt header `User-Agent`.

| Endpoint | Việc |
|---|---|
| `GET /health` | `{"status","queue","running","model","version"}` |
| `POST /v1/asr/jobs` | Tạo job, trả `202` |
| `GET /v1/asr/jobs/{id}` | Trạng thái và kết quả |
| `DELETE /v1/asr/jobs/{id}` | Huỷ job |
| `GET /v1/asr/jobs/{id}/result.vtt` | Phụ đề VTT |

```bash
curl https://asr.daoan.online/v1/asr/jobs \
  -H "Authorization: Bearer $(cat .api_key)" -H "Idempotency-Key: 1181195" \
  -H "Content-Type: application/json" \
  -d '{"audio_url":"https://lessonvid.khaothi.ai/2026/10/1181195/master.m3u8","ref":"1181195","max_seconds":7200}'
```

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
- `asrserver/app.py`: API, hàng đợi, lưu job.
- `asrserver/edge_dns.py`: chọn edge CDN qua DNS khác. Tắt mặc định.
- `tools/try_jobs.py`, `tools/load_test.py`: gửi job thử và thử tải.

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ASR_GPU_UTIL` | `0.45` | Phần VRAM vLLM giữ. Encoder dùng thêm khoảng 3 GB ngoài phần này. |
| `ASR_MAX_ACTIVE` | `6` | Số job tải và chép cùng lúc |
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
