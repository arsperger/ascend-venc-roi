# VENC gotchas — per-frame ROI on Ascend (`acl.himpi`)

Non-obvious failures we hit while driving the Ascend VENC encoder directly from Python, and
what fixes them. Each entry: symptom → cause → fix → snippet.

**Scope:** Orange Pi AI Pro, Ascend 310B4, CANN 7.0.0 (toolkit 7.1.0.3.220), Ubuntu 22.04.
Nothing here has been checked on another CANN release or chip.

Every fix below is implemented in [ascend_encoder.py](../src/ascend_venc_roi/ascend_encoder.py)
or [nv12_converter.py](../src/ascend_venc_roi/nv12_converter.py). Where a claim comes from a
measurement on the board rather than the code, the measurement is given.

---

## 1. `time_ref` must be even and strictly increasing

- **Symptom:** every other frame comes out corrupted.
- **Cause:** a constant `time_ref`, for example 0 for every frame.
- **Fix:** 2, 4, 6, …

```python
"time_ref": (frame_idx + 1) * 2,
```

## 2. `gop_attr` without `normal_p` segfaults `venc_create_chn`

- **Symptom:** the process dies inside `venc_create_chn`. No error code is returned.
- **Cause:** `normal_p` is not documented as required, but the channel cannot be created
  without it.
- **Fix:** always send it.

```python
"gop_attr": {"gop_mode": 0, "normal_p": {"ip_qp_delta": 2}},
```

## 3. Frame memory must be DVPP memory — and that gives you zero-copy

- **Symptom:** VENC does not accept the frame.
- **Cause:** the buffer came from `acl.rt.malloc`.
- **Fix:** allocate with `himpi.dvpp_malloc`. VPC can convert BGR to NV12 straight into a
  DVPP buffer, and that same device address goes into the VENC frame descriptor. The frame
  never returns to host memory between colour conversion and encode.

```python
nv12_addr = himpi.dvpp_malloc(0, nv12_size)            # once
# per frame: VPC writes BGR→NV12 into nv12_addr, then
"virt_addr": [nv12_addr, nv12_addr + y_size, 0],      # VENC reads it in place
```

If you reuse one NV12 buffer, encode the frame before converting the next one.

## 4. At most 8 regions per frame

- **Symptom:** regions past the eighth have no effect.
- **Cause:** the hardware has ROI slots 0–7 only.
- **Fix:** rank your regions before you send them. This library keeps the first 8 in list
  order and drops the rest **without a warning**.

## 5. Rectangles are 16-pixel aligned, and the keys are `x/y/width/height`

- **Symptom:** the ROI is rejected, or lands in the wrong place.
- **Cause:** unaligned coordinates, or `left`/`top` used as keys.
- **Fix:** round position down and size up to 16, clamp to the frame, and skip anything
  smaller than 16×16. See `map_roi_to_venc_rect`.

```python
"rect": {"x": x, "y": y, "width": w, "height": h},
```

## 6. `QP = 28 + qoffset · 18`

This is **this library's mapping**, not a VENC rule. ROI is sent as an absolute QP
(`is_abs_qp: 1`), and `qoffset ∈ [-1, +1]` maps around QP 28.

- **Getting it backwards:** `28 − qoffset·18` makes the "better" regions worse.
- **Why `+`:** negative qoffset means better quality, and lower QP means better quality.

| qoffset | QP |
|---|---|
| −1.0 | 10 |
| −0.5 | 19 |
| 0.0 | 28 |
| +0.5 | 37 |

## 7. VBR clamps ROI QP at 24 by default

- **Symptom:** in VBR, ROI regions with qoffset below about −0.2 look no better than −0.2.
- **Cause:** the channel's default QP floor. Read from a fresh channel on the board:

  | Mode | P/B QP range | I QP range |
  |---|---|---|
  | CBR | 10–51 | 10–51 |
  | VBR | **24**–51 | **24**–51 |

  QP clamping outranks rate control, so a VBR request for QP 19 is encoded at 24.
- **Fix:** after `venc_create_chn` and before `venc_start_chn`, get, change, and set the
  RC params. In this library, `EncoderConfig.qp_min` defaults to 10.

```python
rc, _ = himpi.venc_get_rc_param(chn, 12)             # 12 = H.265 VBR
rc["h265_vbr_param"].update(min_qp=10, min_i_qp=10)
himpi.venc_set_rc_param(chn, rc)
```

## 8. NumPy 2.x: `bytes_to_ptr`, not `numpy_to_ptr`

- **Symptom:** the host-to-device upload fails under NumPy 2.x.
- **Fix:**

```python
host_ptr = acl.util.bytes_to_ptr(frame.tobytes())
acl.rt.memcpy(dvpp_addr, size, host_ptr, size, 1)    # 1 = host → device
```

## 9. `acl.init()` once per process

- **Symptom:** a second `acl.init()` returns `100002`.
- **Fix:** treat `100002` as "already initialised". Only the caller that got `0` should
  finalise. If your detector and the encoder share a process, initialise once outside both
  and pass `manage_acl=False` to the encoder.

## 10. The first frames return no packets

- **Symptom:** `venc_query_status` reports `cur_packs == 0` right after `venc_send_frame`.
- **Fix:** poll briefly. This library retries up to 50 times, 1 ms apart, before counting
  the frame as dropped.

## 11. `vpc_create_chn` needs an empty dict

```python
himpi.vpc_create_chn(vpc_chn, {})
```

---

## Simpler than you may have read

Both of these work in this library across thousands of frames:

- `venc_get_stream` with only `{"pack_cnt": N}`. No pre-allocated pack dict is needed.
- `venc_release_stream` after every frame. You do not have to batch releases.

## Not covered

- **H.264, and H.265 fixed-QP mode (`rc_mode=16`).** This library uses H.265 with CBR (11)
  or VBR (12) only, so we have nothing verified to say about the other modes.
