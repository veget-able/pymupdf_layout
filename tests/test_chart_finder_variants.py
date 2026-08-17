from pathlib import Path

import pytest

from pymupdf.layout import chart_finder


def test_bundled_chart_finder_variants_exist():
    assert set(chart_finder.MODEL_VARIANTS) == {
        "fp32",
        "weight-fp16",
        "mixed-sensitive-fp16",
    }
    for variant in chart_finder.MODEL_VARIANTS:
        path = chart_finder.model_path_for_variant(variant)
        assert isinstance(path, Path)
        assert path.is_file()


def test_chart_finder_variant_validation():
    with pytest.raises(ValueError, match="unknown chart finder variant"):
        chart_finder.model_path_for_variant("unknown")


def test_explicit_model_and_variant_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        chart_finder._resolve_model_path(
            model_path="custom.onnx",
            variant="mixed-sensitive-fp16",
        )


def test_mixed_sensitive_fp16_has_no_separate_enable_gate():
    assert chart_finder._resolve_model_path(
        model_path=None,
        variant="mixed-sensitive-fp16",
    ) == chart_finder.MODEL_VARIANTS["mixed-sensitive-fp16"]


def test_find_charts_preserves_detector_bbox_without_second_detection(monkeypatch):
    calls = {"detect": 0}

    monkeypatch.setattr(
        chart_finder,
        "_resolve_model_path",
        lambda **_kwargs: Path("model.onnx"),
    )
    monkeypatch.setattr(chart_finder, "_session", lambda *_args: object())
    monkeypatch.setattr(chart_finder, "_upright", lambda page: (page, None))

    def detect(_page, _session, _threshold):
        calls["detect"] += 1
        return [[10.0, 10.0, 40.0, 40.0]], [0.9]

    monkeypatch.setattr(chart_finder, "_detect", detect)
    monkeypatch.setattr(
        chart_finder,
        "_refine",
        lambda _page, _boxes, scores: (
            [[8.0, 8.0, 44.0, 44.0]],
            scores,
        ),
    )

    result = chart_finder.find_charts(
        object(),
        variant="fp32",
        include_detector_bbox=True,
    )

    assert calls["detect"] == 1
    assert result == [
        {
            "bbox": (8.0, 8.0, 44.0, 44.0),
            "score": 0.9,
            "detector_bbox": [10.0, 10.0, 40.0, 40.0],
        }
    ]


def test_detector_bbox_matching_is_one_to_one():
    items = [
        {"bbox": [0.0, 0.0, 12.0, 12.0]},
        {"bbox": [20.0, 20.0, 32.0, 32.0]},
    ]

    chart_finder._attach_detector_boxes(
        items,
        [
            [21.0, 21.0, 31.0, 31.0],
            [1.0, 1.0, 11.0, 11.0],
        ],
    )

    assert items[0]["detector_bbox"] == [1.0, 1.0, 11.0, 11.0]
    assert items[1]["detector_bbox"] == [21.0, 21.0, 31.0, 31.0]
