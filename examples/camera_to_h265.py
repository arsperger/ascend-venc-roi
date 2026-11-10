#!/usr/bin/env python3
"""Camera or file → YOLOv8n (.om) → per-frame ROI → H.265.

This is an example, not library API. The detector and the qoffset policy live
here on purpose, outside ``ascend_venc_roi``. The library
receives rectangles and qoffsets only.

Requires CANN (pyACL) and an Ascend NPU.

Example::

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 examples/camera_to_h265.py \\
        --video clip.mp4 --model yolov8n.om --output /tmp/out.h265
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ascend_venc_roi import EncoderConfig, RateControl, ROIRegion
from ascend_venc_roi.ascend_encoder import AscendVENCEncoder

_BASE_QOFFSET = -0.3
_CONFIDENCE_WEIGHT = 0.3
_CONFIDENCE_THRESHOLD = 0.5
_SIZE_WEIGHT = 0.2
_MIN_SIZE_RATIO = 0.01
_MAX_SIZE_RATIO = 0.25
_MARGIN_PX = 10
_MARGIN_RATIO = 0.05
_MAX_REGIONS = 8

_CLASS_OFFSET = {
    "person": -0.1,
    "car": -0.1,
    "bear": -0.1,
    "bird": 0.1,
    "sheep": 0.1,
    "cow": 0.1,
}

_COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard",
    "surfboard", "tennis_racket", "bottle", "wine_glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair",
    "couch", "potted_plant", "bed", "dining_table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell_phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy_bear", "hair_drier", "toothbrush",
)

_ACL_H2D = 1
_ACL_D2H = 2
_ACL_MALLOC_NORMAL = 2


def qoffset_for(
    class_name: str,
    confidence: float,
    box_area: int,
    frame_area: int,
) -> float:
    """Relative quality offset in ``[-1, 0]`` for one detection."""
    qoffset = _BASE_QOFFSET + _CLASS_OFFSET.get(class_name, 0.0)
    if confidence >= _CONFIDENCE_THRESHOLD:
        confidence_factor = 1.0
    elif confidence <= 0.0:
        confidence_factor = 0.0
    else:
        confidence_factor = confidence / _CONFIDENCE_THRESHOLD
    qoffset *= 1.0 - _CONFIDENCE_WEIGHT * (1.0 - confidence_factor)

    ratio = box_area / frame_area if frame_area else 1.0
    if ratio <= _MIN_SIZE_RATIO:
        size_factor = -1.0
    elif ratio >= _MAX_SIZE_RATIO:
        size_factor = 1.0
    else:
        span = _MAX_SIZE_RATIO - _MIN_SIZE_RATIO
        size_factor = 2.0 * (ratio - _MIN_SIZE_RATIO) / span - 1.0
    qoffset *= 1.0 - _SIZE_WEIGHT * size_factor
    return max(-1.0, min(0.0, qoffset))


def regions_for_detections(
    detections: list[tuple[int, int, int, int, str, float]],
    frame_width: int,
    frame_height: int,
) -> list[ROIRegion]:
    """Turn ``(x, y, w, h, class_name, confidence)`` rows into at most 8 regions."""
    frame_area = frame_width * frame_height
    ranked: list[tuple[float, float, ROIRegion]] = []
    for x, y, w, h, name, confidence in detections:
        if w <= 0 or h <= 0:
            continue
        margin_x = max(_MARGIN_PX, int(w * _MARGIN_RATIO))
        margin_y = max(_MARGIN_PX, int(h * _MARGIN_RATIO))
        nx = max(0, x - margin_x)
        ny = max(0, y - margin_y)
        nw = min(frame_width - nx, w + 2 * margin_x)
        nh = min(frame_height - ny, h + 2 * margin_y)
        qoffset = qoffset_for(name, confidence, w * h, frame_area)
        ranked.append((
            _CLASS_OFFSET.get(name, 0.0),
            -confidence,
            ROIRegion(nx, ny, nw, nh, qoffset),
        ))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [region for _, _, region in ranked[:_MAX_REGIONS]]


class YoloOmDetector:
    """Minimal YOLOv8n ``.om`` runner. Output layout is ``(1, 84, 8400)``."""

    def __init__(self, model_path: Path, conf_threshold: float) -> None:
        import acl

        self._acl = acl
        self._conf = conf_threshold
        self._model_id, ret = acl.mdl.load_from_file(str(model_path))
        if ret != 0:
            raise RuntimeError(f"acl.mdl.load_from_file failed: {ret}")
        self._desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._desc, self._model_id)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_desc failed: {ret}")
        self._context, ret = acl.rt.get_context()
        if ret != 0:
            self._context, ret = acl.rt.create_context(0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.create_context failed: {ret}")

        self._in_size = acl.mdl.get_input_size_by_index(self._desc, 0)
        self._out_size = acl.mdl.get_output_size_by_index(self._desc, 0)
        self._in_buf, ret = acl.rt.malloc(self._in_size, _ACL_MALLOC_NORMAL)
        if ret != 0:
            raise RuntimeError(f"input malloc failed: {ret}")
        self._out_buf, ret = acl.rt.malloc(self._out_size, _ACL_MALLOC_NORMAL)
        if ret != 0:
            raise RuntimeError(f"output malloc failed: {ret}")
        self._in_ds = acl.mdl.create_dataset()
        self._out_ds = acl.mdl.create_dataset()
        _, ret = acl.mdl.add_dataset_buffer(
            self._in_ds, acl.create_data_buffer(self._in_buf, self._in_size),
        )
        if ret != 0:
            raise RuntimeError(f"add input buffer failed: {ret}")
        _, ret = acl.mdl.add_dataset_buffer(
            self._out_ds, acl.create_data_buffer(self._out_buf, self._out_size),
        )
        if ret != 0:
            raise RuntimeError(f"add output buffer failed: {ret}")

    def detect(
        self, frame: npt.NDArray[np.uint8],
    ) -> list[tuple[int, int, int, int, str, float]]:
        """Return detections as ``(x, y, w, h, class_name, confidence)``."""
        acl = self._acl
        ret = acl.rt.set_context(self._context)
        if ret != 0:
            raise RuntimeError(f"set_context failed: {ret}")

        height, width = frame.shape[:2]
        tensor = _preprocess(frame)
        if tensor.nbytes != self._in_size:
            raise RuntimeError(
                f"model input is {self._in_size} bytes, tensor is {tensor.nbytes}",
            )
        host = tensor.__array_interface__["data"][0]
        ret = acl.rt.memcpy(self._in_buf, self._in_size, host, tensor.nbytes, _ACL_H2D)
        if ret != 0:
            raise RuntimeError(f"H2D memcpy failed: {ret}")
        ret = acl.mdl.execute(self._model_id, self._in_ds, self._out_ds)
        if ret != 0:
            raise RuntimeError(f"mdl.execute failed: {ret}")

        output = np.zeros(self._out_size // 4, dtype=np.float32)
        dest = output.__array_interface__["data"][0]
        ret = acl.rt.memcpy(dest, self._out_size, self._out_buf, self._out_size, _ACL_D2H)
        if ret != 0:
            raise RuntimeError(f"D2H memcpy failed: {ret}")
        return _decode(output, width, height, self._conf)

    def close(self) -> None:
        acl = self._acl
        acl.rt.free(self._in_buf)
        acl.rt.free(self._out_buf)
        acl.mdl.unload(self._model_id)
        acl.mdl.destroy_desc(self._desc)


def _preprocess(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    import cv2

    image = cv2.resize(frame, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = image.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor[None, ...])


def _decode(
    output: npt.NDArray[np.float32],
    frame_width: int,
    frame_height: int,
    conf_threshold: float,
) -> list[tuple[int, int, int, int, str, float]]:
    values = 4 + len(_COCO_NAMES)
    if output.size % values != 0:
        raise RuntimeError(f"unexpected YOLO output size {output.size}")
    preds = output.reshape(1, values, -1)[0].T
    boxes = preds[:, :4]
    class_scores = preds[:, 4:]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    keep = scores >= conf_threshold
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    if len(boxes) == 0:
        return []

    scale_x = frame_width / 640
    scale_y = frame_height / 640
    x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2) * scale_x, 0, frame_width)
    y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2) * scale_y, 0, frame_height)
    x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2) * scale_x, 0, frame_width)
    y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2) * scale_y, 0, frame_height)
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    chosen = _nms(xyxy, scores, 0.45)

    detections: list[tuple[int, int, int, int, str, float]] = []
    for index in chosen:
        left, top, right, bottom = (int(v) for v in xyxy[index])
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue
        name = _COCO_NAMES[int(class_ids[index])]
        detections.append((left, top, width, height, name, float(scores[index])))
    return detections


def _nms(
    boxes: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while len(order) > 0:
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / (area_i + area_rest - intersection + 1e-6)
        order = rest[iou <= iou_threshold]
    return kept


def _init_runtime() -> None:
    """acl + HiMPI once, before the detector and the encoder share the device."""
    import acl
    from acl import himpi

    ret = acl.init()
    if ret not in (0, 100002):
        raise RuntimeError(f"acl.init() failed: {ret}")
    ret = acl.rt.set_device(0)
    if ret != 0:
        raise RuntimeError(f"set_device(0) failed: {ret}")
    ret = himpi.sys_init()
    if ret not in (0, 0xA0058008):
        raise RuntimeError(f"himpi.sys_init() failed: 0x{ret:08x}")


def _open_capture(video: str | None, camera: str | None) -> tuple[object, int, int, float]:
    import cv2

    source: str | int = 0 if camera is None else camera
    if video is not None:
        source = video
    elif camera is not None and camera.isdigit():
        source = int(camera)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0:
        fps = 30.0
    return capture, width, height, fps


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Input video file")
    source.add_argument("--camera", help="V4L2 device, e.g. 0 or /dev/video0")
    parser.add_argument("--model", required=True, help="YOLOv8n .om model")
    parser.add_argument("--output", required=True, help="Output .h265 path")
    parser.add_argument("--bitrate-kbps", type=int, default=2000)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 encodes the whole input")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    model = Path(args.model)
    if not model.is_file():
        print(f"model not found: {model}", file=sys.stderr)
        return 1

    _init_runtime()
    detector = YoloOmDetector(model, args.conf)
    capture, width, height, fps = _open_capture(args.video, args.camera)
    if width % 16 or height % 16:
        print(f"frame size {width}x{height} is not 16-pixel aligned", file=sys.stderr)
        return 1

    config = EncoderConfig(
        output_path=args.output,
        width=width,
        height=height,
        fps=fps,
        bitrate_kbps=args.bitrate_kbps,
        rate_control=RateControl.CBR,
        gop=30,
    )
    # Runtime was initialised above. The encoder must not call acl.finalize().
    encoder = AscendVENCEncoder(config, manage_acl=False)
    started = time.perf_counter()
    frames = 0
    detections = 0
    try:
        if not encoder.open():
            print("encoder failed to open", file=sys.stderr)
            return 1
        while True:
            if args.max_frames and frames >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            found = detector.detect(frame)
            detections += len(found)
            regions = regions_for_detections(found, width, height)
            if not encoder.write_frame(frame, regions):
                print(f"write_frame failed at frame {frames}", file=sys.stderr)
                return 1
            frames += 1
    finally:
        encoder.close()
        detector.close()
        capture.release()

    elapsed = time.perf_counter() - started
    print(
        f"frames={frames} detections={detections} "
        f"elapsed_s={elapsed:.1f} fps={frames / elapsed if elapsed else 0:.1f} "
        f"output={args.output}",
    )
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
