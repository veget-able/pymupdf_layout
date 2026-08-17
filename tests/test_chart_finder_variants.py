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
