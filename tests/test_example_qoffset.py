"""Off-device checks for the example's inlined qoffset policy."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_to_h265.py"
_SPEC = importlib.util.spec_from_file_location("camera_to_h265", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def test_person_gets_a_stronger_boost_than_an_unknown_class() -> None:
    person = example.qoffset_for("person", 0.9, 200 * 400, 1280 * 720)
    other = example.qoffset_for("toaster", 0.9, 200 * 400, 1280 * 720)
    assert person < other
    assert -1.0 <= person <= 0.0


def test_low_confidence_damps_the_boost() -> None:
    high = example.qoffset_for("car", 0.9, 100 * 100, 1280 * 720)
    low = example.qoffset_for("car", 0.1, 100 * 100, 1280 * 720)
    assert high < low


def test_regions_are_capped_at_eight() -> None:
    rows = [(10 * i, 10, 40, 40, "person", 0.9) for i in range(12)]
    regions = example.regions_for_detections(rows, 1280, 720)
    assert len(regions) == 8
