from pathlib import Path

import pytest

from pymupdf.layout import chart_picture_finder


def test_bundled_chart_picture_variants_exist():
    assert set(chart_picture_finder.MODEL_VARIANTS) == {
        "fp32",
        "weight-fp16",
        "mixed-sensitive-fp16",
    }
    for variant in chart_picture_finder.MODEL_VARIANTS:
        path = chart_picture_finder.model_path_for_variant(variant)
        assert isinstance(path, Path)
        assert path.is_file()


def test_mixed_sensitive_fp16_has_no_separate_enable_gate():
    assert chart_picture_finder._resolve_model_path(
        model_path=None,
        variant="mixed-sensitive-fp16",
    ) == chart_picture_finder.MODEL_VARIANTS["mixed-sensitive-fp16"]


def test_explicit_model_and_variant_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        chart_picture_finder._resolve_model_path(
            model_path="custom.onnx",
            variant="weight-fp16",
        )


def test_picture_refiner_removes_only_multichild_parent():
    boxes, scores, stats = (
        chart_picture_finder._suppress_multichild_picture_parents(
            [
                [0, 0, 100, 100],
                [0, 0, 100, 45],
                [0, 55, 100, 100],
            ],
            [0.9, 0.8, 0.7],
        )
    )
    assert boxes == [[0.0, 0.0, 100.0, 45.0], [0.0, 55.0, 100.0, 100.0]]
    assert scores == [0.8, 0.7]
    assert stats["parents_removed"] == 1


def test_picture_refiner_preserves_single_containment_pair():
    boxes, scores, stats = (
        chart_picture_finder._suppress_multichild_picture_parents(
            [[0, 0, 100, 100], [20, 20, 80, 80]],
            [0.9, 0.8],
        )
    )
    assert len(boxes) == 2
    assert scores == [0.9, 0.8]
    assert stats["parents_removed"] == 0


def test_dp0_preserves_chart_detector_bbox_without_second_detection(monkeypatch):
    calls = {"detect": 0}

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        chart_picture_finder,
        "_resolve_model_path",
        lambda **_kwargs: Path("model.onnx"),
    )
    monkeypatch.setattr(
        chart_picture_finder.chart_finder,
        "_session",
        lambda *_args: Session(),
    )
    monkeypatch.setattr(
        chart_picture_finder.chart_finder,
        "_upright",
        lambda page: (page, None),
    )

    def detect(_page, _session, _thresholds):
        calls["detect"] += 1
        return {
            "chart": {
                "boxes": [[10.0, 10.0, 40.0, 40.0]],
                "scores": [0.9],
            },
            "picture": {
                "boxes": [[50.0, 50.0, 70.0, 70.0]],
                "scores": [0.8],
            },
        }

    monkeypatch.setattr(chart_picture_finder, "_detect", detect)
    monkeypatch.setattr(
        chart_picture_finder.chart_finder,
        "_refine",
        lambda _page, _boxes, scores: (
            [[8.0, 8.0, 44.0, 44.0]],
            scores,
        ),
    )

    result = chart_picture_finder.find_chart_pictures(
        object(),
        variant="fp32",
        include_detector_bbox=True,
    )

    assert calls["detect"] == 1
    assert result["chart"][0]["bbox"] == [8.0, 8.0, 44.0, 44.0]
    assert result["chart"][0]["detector_bbox"] == [10.0, 10.0, 40.0, 40.0]
    assert result["picture"][0]["bbox"] == [50.0, 50.0, 70.0, 70.0]
