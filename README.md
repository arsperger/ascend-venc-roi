# ascend-venc-roi

Per-frame Region-of-Interest H.265 encoding on Huawei Ascend NPUs, via the native VENC API
(`acl.himpi`). You give it a frame and a list of rectangles with quality offsets; it updates
the encoder's QP map **every frame** — about 27 FPS at 1280×720 with YOLOv8n detection on an
Orange Pi AI Pro.

## When ROI is worth it — and when it isn't

ROI is **not** a universal win. Of 18 scene × resolution cells we measured, ROI is
**recommended in 6, conditional in 1, and the wrong choice in 11.** It pays on static scenes
and on a camera tracking its target, at HD and FHD. Elsewhere the gain inside objects is
under 0.1 dB, or the encoder overshoots the bitrate target 4–38× once ROI is on.

Each cell: verdict · PSNR gain inside detected objects (ΔPSNR_ROI) · worst bitrate overshoot
(actual ÷ target).

| Scene | HD | FHD | 4K |
|---|---|---|---|
| S1 sparse, static camera | **ROI** · +0.35 dB · 0.98× | **ROI** · +0.35 dB · 0.93× | conditional · +0.10 dB · 1.39× |
| S2 dense, static camera | **ROI** · +0.32 dB · 1.42× | **ROI** · +0.30 dB · 1.21× | baseline · +0.07 dB · 1.34× |
| S3 sparse, moving objects | baseline · +0.79 dB · 14.7× | baseline · +1.18 dB · 28.1× | baseline · +0.30 dB · 37.9× |
| S4 dense, moving objects | baseline · +0.04 dB · 1.01× | baseline · −0.05 dB · 1.36× | baseline · +0.01 dB · 4.50× |
| S5 PTZ scanning | baseline · +0.09 dB · 1.24× | baseline · +0.19 dB · 10.5× | baseline · +0.05 dB · 21.3× |
| S6 PTZ tracking a target | **ROI** · +1.90 dB · 1.25× | **ROI** · +2.20 dB · 1.27× | baseline · +0.75 dB · 4.39× |

- **ROI** — gain ≥ 0.20 dB and overshoot ≤ 1.5×.
- **conditional** — gain 0.10–0.20 dB, or overshoot 1.5–2.0×.
- **baseline** — everything else. S3 gains inside objects, but at 14.7–37.9× the bitrate
  you asked for, which defeats the point of ROI.

**What to expect.** Across all scenes, the median gain inside objects is **+0.249 dB** PSNR
(paired ROI vs. plain encoding at the same target, n = 900). That median hides a split: when
the encoder overshoots the bitrate target (> 1.10×) the median is +0.57 dB, and when it stays
on target it is **+0.12 dB**. If your link enforces the bitrate, plan for +0.12 dB. The best
cell, S6 at FHD, reaches +2.20 dB.

### How this was measured

1006 encoding runs on an Orange Pi AI Pro (Ascend 310B4, CANN 7.0.0). Six scenes (S1–S6) at
HD, FHD and 4K; VBR, GOP 30, targets of 1–8 Mbit/s (4K also 12–20 Mbit/s); YOLOv8n
detection. Every ROI run is paired with a plain encode of the same clip at the same target.
Quality is PSNR inside the union of detected boxes. In each cell we tried 10 ROI settings
(qoffset −0.1 … −0.5, background offset 0 or 0.3) and kept the best one whose bitrate stayed
within 1.5× of target; if none did, the cell shows the best one regardless.

## Why this exists

`ffmpeg -vf addroi` only works with `libx265` (~7 FPS on this class of device). The hardware
encoder `h265_ascend` is fast (~68 FPS) but ignores ROI side data. This library takes the
third path: drive VENC directly and set the ROI attributes per frame.

Doing that from Python runs into undocumented constraints — a missing field that segfaults,
a timestamp rule that corrupts every other frame, a default QP floor that hides ROI in VBR.
They are written up in **[docs/venc-gotchas.md](docs/venc-gotchas.md)**.

## Quickstart

Verified toolchain on the reference board: CANN 7.0.0, toolkit `7.1.0.3.220`
(`/usr/local/Ascend/ascend-toolkit/latest/version.cfg`). Do not skip the environment script —
without it `import acl` fails.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -e ".[example]"
python3 examples/camera_to_h265.py \
    --video clip.mp4 \
    --model yolov8n.om \
    --output /tmp/out.h265
```

`--model` is a YOLOv8n model already converted to `.om` (ATC, input `images`, 1×3×640×640).
The script does not download or convert a model. Use `--camera 0` instead of `--video` for a
V4L2 camera.

## Using the library

The library takes rectangles and quality offsets, not detections — bring your own detector
and policy. The example shows one.

```python
from ascend_venc_roi import EncoderConfig, RateControl, ROIRegion
from ascend_venc_roi.ascend_encoder import AscendVENCEncoder

cfg = EncoderConfig(output_path="out.h265", width=1280, height=720,
                    bitrate_kbps=2000, rate_control=RateControl.VBR)
with AscendVENCEncoder(cfg) as enc:
    for frame, boxes in stream:                      # BGR uint8 frames
        enc.write_frame(frame, [ROIRegion(x, y, w, h, qoffset=-0.3) for x, y, w, h in boxes])
```

`qoffset` runs from −1.0 (best quality) to +1.0. At most 8 regions per frame are used.

## Requirements

- Ascend NPU (verified: 310B4 / Orange Pi AI Pro)
- CANN 7.0.0 with pyACL (toolkit `7.1.0.3.220` on the reference board)
- Python 3.10+
- Ubuntu 22.04

A Jetson / NVENC backend is not built, but the `IEncoder` interface is kept open for one.

## License

Apache-2.0
