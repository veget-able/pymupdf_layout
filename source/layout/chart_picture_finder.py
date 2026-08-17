"""Two-class Chart/Picture detection with shared chart refinement.

This module bundles the DP0 FP32, weight-FP16 and mixed-sensitive-FP16 ONNX
variants. The mixed-precision model keeps sensitive reductions in FP32 while
using FP16 convolutions and weights. Chart detections use the same refinement
implementation as ``chart_finder``. Picture detections retain their detector
geometry, except that a proposal containing two or more smaller Picture
proposals is suppressed as an oversized detector parent. A single containment
relation is preserved.
"""

from pathlib import Path

import numpy as np

from . import chart_finder


__version__ = '260812'

MODEL_PATH = (
    Path(__file__).resolve().parent
    / 'resources' / 'onnx' / 'chart_picture_finder.onnx'
)
MODEL_VARIANTS = {
    'fp32': MODEL_PATH,
    'weight-fp16': MODEL_PATH.with_name(
        'chart_picture_finder_weight_fp16.onnx'),
    'mixed-sensitive-fp16': MODEL_PATH.with_name(
        'chart_picture_finder_mixed_sensitive_fp16.onnx'),
}
CLASS_NAMES = {0: 'chart', 1: 'picture'}
MAX_DETECTIONS_PER_CLASS = 100
PICTURE_CONTAIN_OVERLAP = 0.90
PICTURE_PARENT_MIN_CHILDREN = 2


def find_chart_pictures(
        page,
        *,
        chart_threshold=0.5,
        picture_threshold=0.5,
        providers=None,
        device_id=0,
        model_path=None,
        variant=None,
):
    """Detect and refine Chart/Picture regions on one page.

    ``variant`` accepts ``fp32`` (the default), ``weight-fp16`` or
    ``mixed-sensitive-fp16``. An explicit ``model_path`` is also accepted and
    is mutually exclusive with ``variant``. No precision variant requires a
    separate enable flag.

    Returns a dictionary with separate ``chart`` and ``picture`` lists. Each
    list item contains ``bbox``, ``score`` and ``label``.
    """
    thresholds = {
        0: _threshold(chart_threshold, 'chart_threshold'),
        1: _threshold(picture_threshold, 'picture_threshold'),
    }
    if device_id < 0:
        raise ValueError('device_id must be non-negative')

    selected_variant = variant or ('custom' if model_path is not None else 'fp32')
    model_path = _resolve_model_path(model_path=model_path, variant=variant)
    session = chart_finder._session(model_path, providers, device_id)
    page, owner = chart_finder._upright(page)
    try:
        raw = _detect(page, session, thresholds)
        chart_boxes, chart_scores = chart_finder._refine(
            page,
            raw['chart']['boxes'],
            raw['chart']['scores'],
        )
        picture_boxes, picture_scores, picture_refinement = (
            _suppress_multichild_picture_parents(
                raw['picture']['boxes'],
                raw['picture']['scores'],
            )
        )
        return {
            'chart': _items('chart', chart_boxes, chart_scores),
            'picture': _items('picture', picture_boxes, picture_scores),
            'model_variant': selected_variant,
            'model_path': str(model_path),
            'providers': list(session.get_providers()),
            'refinement': {
                'chart': 'chart_finder._refine',
                'picture': picture_refinement,
            },
        }
    finally:
        if owner is not None:
            owner.close()


def model_path_for_variant(variant='fp32'):
    """Return the bundled ONNX path for a named precision variant."""
    try:
        path = MODEL_VARIANTS[str(variant)]
    except KeyError as exc:
        raise ValueError(
            f'unknown chart/picture finder variant {variant!r}; '
            f'expected one of {sorted(MODEL_VARIANTS)}') from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_model_path(*, model_path, variant):
    if model_path is not None and variant is not None:
        raise ValueError('model_path and variant are mutually exclusive')
    if model_path is not None:
        return Path(model_path).expanduser().resolve()
    return model_path_for_variant(variant or 'fp32')


def _threshold(value, name):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1]')
    return value


def _detect(page, session, thresholds):
    tensor, scale_factor, render_scale = chart_finder._render(page)
    try:
        rows, counts = session.run(
            None,
            {'image': tensor, 'scale_factor': scale_factor},
        )
    except Exception as exc:
        message = str(exc)
        if 'Gather' not in message or 'axis 0' not in message:
            raise
        rows = np.empty((0, 6), dtype=np.float32)
        counts = np.array([0], dtype=np.int32)

    count = min(int(np.asarray(counts).reshape(-1)[0]), len(rows))
    boxes = {class_id: [] for class_id in CLASS_NAMES}
    scores = {class_id: [] for class_id in CLASS_NAMES}
    for row in sorted(
            rows[:count].tolist(), key=lambda item: float(item[1]), reverse=True):
        class_id = int(row[0])
        score = float(row[1])
        if class_id not in CLASS_NAMES or score < thresholds[class_id]:
            continue
        if len(boxes[class_id]) >= MAX_DETECTIONS_PER_CLASS:
            continue
        box = chart_finder._page_box(row[2:6], render_scale, page.rect)
        if not box:
            continue
        boxes[class_id].append(box)
        scores[class_id].append(round(score, 6))

    return {
        class_name: {
            'boxes': boxes[class_id],
            'scores': scores[class_id],
        }
        for class_id, class_name in CLASS_NAMES.items()
    }


def _items(label, boxes, scores):
    return [
        {
            'bbox': [round(float(value), 4) for value in box],
            'score': float(scores[index]),
            'label': label,
            'source': 'chart_picture_finder',
        }
        for index, box in enumerate(boxes)
    ]


def _suppress_multichild_picture_parents(
        boxes,
        scores,
        *,
        contain_overlap=PICTURE_CONTAIN_OVERLAP,
        min_children=PICTURE_PARENT_MIN_CHILDREN,
):
    """Remove only a Picture proposal containing multiple smaller proposals."""
    if len(boxes) != len(scores):
        raise ValueError('Picture boxes and scores must have the same length')
    contain_overlap = float(contain_overlap)
    if not 0.0 <= contain_overlap <= 1.0:
        raise ValueError('contain_overlap must be in [0, 1]')
    if min_children < 2:
        raise ValueError('min_children must be at least 2')

    boxes = [[float(value) for value in box] for box in boxes]
    scores = [float(value) for value in scores]
    areas = [_area(box) for box in boxes]
    children_by_parent = {}
    for parent_index, parent in enumerate(boxes):
        parent_area = areas[parent_index]
        if parent_area <= 0:
            continue
        children = []
        for child_index, child in enumerate(boxes):
            child_area = areas[child_index]
            if child_index == parent_index or not 0 < child_area < parent_area:
                continue
            if _intersection_area(parent, child) / child_area >= contain_overlap:
                children.append(child_index)
        if len(children) >= min_children:
            children_by_parent[parent_index] = children

    suppressed = set(children_by_parent)
    retained = [index for index in range(len(boxes)) if index not in suppressed]
    return (
        [boxes[index] for index in retained],
        [scores[index] for index in retained],
        {
            'mode': 'multi-child-parent-suppression',
            'contain_overlap': contain_overlap,
            'min_children': min_children,
            'parents_removed': len(suppressed),
            'parent_child_links': sum(
                len(children) for children in children_by_parent.values()),
            'input_boxes': len(boxes),
            'output_boxes': len(retained),
        },
    )


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first, second):
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1]))
