"""ROI slot bookkeeping in AscendVENCEncoder, with a fake ``acl.himpi``."""

import sys
import types
from typing import Any

import pytest

from ascend_venc_roi import EncoderConfig, ROIRegion, ascend_encoder
from ascend_venc_roi.ascend_encoder import AscendVENCEncoder


class FakeHimpi:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def venc_set_roi_attr(self, chn: int, attr: dict[str, Any]) -> int:
        self.calls.append(attr)
        return 0

    def take(self) -> dict[int, int]:
        """idx → enable for the calls since the last take()."""
        out = {c["idx"]: c["enable"] for c in self.calls}
        self.calls.clear()
        return out


@pytest.fixture
def himpi(monkeypatch: pytest.MonkeyPatch) -> FakeHimpi:
    fake = FakeHimpi()
    acl = types.ModuleType("acl")
    acl.himpi = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acl", acl)
    return fake


@pytest.fixture
def encoder() -> AscendVENCEncoder:
    return AscendVENCEncoder(EncoderConfig(output_path="/dev/null", width=1280, height=720))


def _regions(n: int) -> list[ROIRegion]:
    return [ROIRegion(64 * i, 64, 64, 64, -0.5) for i in range(n)]


class TestStaleROISlots:
    def test_fewer_regions_disable_the_leftover_slots(
        self, encoder: AscendVENCEncoder, himpi: FakeHimpi,
    ) -> None:
        encoder._apply_roi_regions(_regions(3))
        assert himpi.take() == {0: 1, 1: 1, 2: 1}

        encoder._apply_roi_regions(_regions(1))
        assert himpi.take() == {0: 1, 1: 0, 2: 0}

    @pytest.mark.parametrize("empty", [[], None])
    def test_no_regions_disable_every_enabled_slot(
        self, encoder: AscendVENCEncoder, himpi: FakeHimpi, empty: list[ROIRegion] | None,
    ) -> None:
        encoder._apply_roi_regions(_regions(2))
        himpi.take()

        encoder._apply_roi_regions(empty)
        assert himpi.take() == {0: 0, 1: 0}

    def test_nothing_to_clear_makes_no_calls(
        self, encoder: AscendVENCEncoder, himpi: FakeHimpi,
    ) -> None:
        encoder._apply_roi_regions([])
        encoder._apply_roi_regions(None)
        assert himpi.calls == []

    def test_slot_of_a_skipped_region_is_disabled(
        self,
        encoder: AscendVENCEncoder,
        himpi: FakeHimpi,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        encoder._apply_roi_regions(_regions(3))
        himpi.take()

        real_map = ascend_encoder.map_roi_to_venc_rect
        skip = ROIRegion(0, 0, 8, 8, -0.5)
        monkeypatch.setattr(
            ascend_encoder,
            "map_roi_to_venc_rect",
            lambda roi, w, h: None if roi is skip else real_map(roi, w, h),
        )
        regions = _regions(3)
        regions[1] = skip
        encoder._apply_roi_regions(regions)
        assert himpi.take() == {0: 1, 1: 0, 2: 1}

    def test_disable_call_is_a_valid_roi_attr(
        self, encoder: AscendVENCEncoder, himpi: FakeHimpi,
    ) -> None:
        encoder._apply_roi_regions(_regions(1))
        himpi.take()
        encoder._apply_roi_regions([])

        (attr,) = himpi.calls
        assert attr["enable"] == 0
        assert 0 <= attr["qp"] <= 51
        assert attr["rect"]["width"] % 16 == attr["rect"]["height"] % 16 == 0
