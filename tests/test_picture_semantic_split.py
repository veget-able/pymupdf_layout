import numpy as np
import pytest

try:
    from pymupdf.layout.common_util import add_picture_semantic_child_groups
except ImportError:
    # Source-tree execution before the wheel installs ``layout`` below the
    # ``pymupdf`` namespace.
    from layout.common_util import add_picture_semantic_child_groups


CLASS_NAMES = ["text", "picture", "section-header", "list-item", "table", "caption"]
PRIORITY = [1, 2, 0, 3, 4, 5]


def _inputs(text_probability=0.8, section_probability=0.7):
    node_cls = np.array([1, 1, 0, 2], dtype=np.int64)
    node_score = np.array([0.9, 0.8, text_probability, section_probability])
    node_probabilities = np.array(
        [
            [0.02, 0.90, 0.02, 0.02, 0.04, 0.00],
            [0.05, 0.80, 0.05, 0.05, 0.05, 0.00],
            [text_probability, 0.10, 0.05, 0.04, 0.01, 0.00],
            [0.10, 0.10, section_probability, 0.05, 0.05, 0.00],
        ],
        dtype=np.float32,
    )
    edge_matrix = np.ones((4, 4), dtype=np.int64) - np.eye(4, dtype=np.int64)
    bboxes = np.array(
        [[0, 0, 10, 10], [12, 0, 22, 10], [0, 12, 22, 22], [0, 24, 22, 34]],
        dtype=np.float32,
    )
    groups = [
        {
            "indicies": [0, 1, 2, 3],
            "group_class": 1,
            "group_bbox": [0.0, 0.0, 22.0, 34.0],
            "tie_class": -1,
            "class_name": "picture",
        }
    ]
    return groups, node_cls, node_score, node_probabilities, edge_matrix, bboxes


def _split(profile, **probabilities):
    values = _inputs(**probabilities)
    return add_picture_semantic_child_groups(
        groups=values[0],
        node_cls=values[1],
        node_score=values[2],
        node_probabilities=values[3],
        edge_matrix=values[4],
        bboxes=values[5],
        label_priority_list=PRIORITY,
        class_names=CLASS_NAMES,
        profile=profile,
    )


def test_confident_profile_retains_parent_and_adds_textual_children():
    groups = _split("confident-text-section")

    assert [group["class_name"] for group in groups] == [
        "picture",
        "text",
        "section-header",
    ]
    assert groups[0]["indicies"] == [0, 1, 2, 3]
    assert groups[1]["indicies"] == [2]
    assert groups[2]["indicies"] == [3]
    assert groups[1]["semantic_parent_group_index"] == 0
    assert groups[1]["semantic_family"] == "text"
    assert groups[2]["semantic_family"] == "section"


def test_confident_profile_filters_child_below_probability_half():
    groups = _split("confident-text-section", text_probability=0.49)

    assert [group["class_name"] for group in groups] == [
        "picture",
        "section-header",
    ]


def test_section_profile_does_not_emit_text_family():
    groups = _split("section-only")

    assert [group["class_name"] for group in groups] == [
        "picture",
        "section-header",
    ]


def test_same_family_edges_still_define_separate_children():
    values = list(_inputs())
    values[1] = np.array([1, 1, 0, 0], dtype=np.int64)
    values[3][3] = np.array([0.75, 0.10, 0.05, 0.05, 0.05, 0.00])
    values[4][2, 3] = values[4][3, 2] = 0
    groups = add_picture_semantic_child_groups(
        groups=values[0],
        node_cls=values[1],
        node_score=values[2],
        node_probabilities=values[3],
        edge_matrix=values[4],
        bboxes=values[5],
        label_priority_list=PRIORITY,
        class_names=CLASS_NAMES,
        profile="confident-text-section",
    )

    assert [group["indicies"] for group in groups[1:]] == [[2], [3]]


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="picture_semantic_split_profile"):
        _split("unknown")


def test_core_profile_excludes_caption_children():
    values = list(_inputs())
    values[1][3] = CLASS_NAMES.index("caption")
    values[3][3] = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.75])
    groups = add_picture_semantic_child_groups(
        groups=values[0],
        node_cls=values[1],
        node_score=values[2],
        node_probabilities=values[3],
        edge_matrix=values[4],
        bboxes=values[5],
        label_priority_list=PRIORITY,
        class_names=CLASS_NAMES,
        profile="confident-core-text-section",
    )

    assert [group["class_name"] for group in groups] == ["picture", "text"]
