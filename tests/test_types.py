"""Tests for the vendor-neutral ROI primitives."""

import ast
from pathlib import Path

from ascend_venc_roi import QOFFSET_MAX, QOFFSET_MIN, ROIRegion


class TestROIRegion:
    def test_fields_are_preserved(self) -> None:
        roi = ROIRegion(x=100, y=50, w=200, h=300, qoffset=-0.3)
        assert (roi.x, roi.y, roi.w, roi.h) == (100, 50, 200, 300)
        assert roi.qoffset == -0.3

    def test_qoffset_clamped_below_range(self) -> None:
        assert ROIRegion(0, 0, 16, 16, qoffset=-4.2).qoffset == QOFFSET_MIN

    def test_qoffset_clamped_above_range(self) -> None:
        assert ROIRegion(0, 0, 16, 16, qoffset=7.5).qoffset == QOFFSET_MAX

    def test_qoffset_boundaries_unchanged(self) -> None:
        assert ROIRegion(0, 0, 16, 16, qoffset=-1.0).qoffset == -1.0
        assert ROIRegion(0, 0, 16, 16, qoffset=1.0).qoffset == 1.0
        assert ROIRegion(0, 0, 16, 16, qoffset=0.0).qoffset == 0.0

    def test_repr_includes_geometry_and_offset(self) -> None:
        assert repr(ROIRegion(1, 2, 3, 4, -0.5)) == "ROIRegion(1,2 3x4 qoffset=-0.500)"


class TestNoDetectorCoupling:
    """The package must not carry detector or FFmpeg code."""

    def test_roi_module_imports_nothing_vendor_or_detector_specific(self) -> None:
        """``ROIRegion`` must not import from ``detection/`` or a vendor SDK.

        Checked against the parsed import graph rather than the raw text — a
        substring scan matches innocent words such as ``dataclass``.
        """
        import ascend_venc_roi.types as types_mod

        source = Path(types_mod.__file__).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        forbidden_roots = ("acl", "cv2", "detection", "roi_encoder", "subprocess")
        for module in imported:
            root = module.split(".")[0]
            assert root not in forbidden_roots, f"types.py must not import {module}"

    def test_dropped_symbols_are_absent_from_public_surface(self) -> None:
        import ascend_venc_roi as pkg

        for dropped in (
            "QPCalculator",
            "QPConfig",
            "ROIPriority",
            "ROIFilterBuilder",
            "FFmpegPipeEncoder",
            "ObjectClass",
        ):
            assert not hasattr(pkg, dropped)

    def test_root_namespace_is_vendor_neutral(self) -> None:
        import ascend_venc_roi as pkg

        assert all("ascend" not in name.lower() for name in pkg.__all__)
