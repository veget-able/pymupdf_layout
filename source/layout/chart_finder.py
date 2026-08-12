"""
chart_finder.py

Chart region detection for a single PDF page.

The detector is a single-class chart detection model trained in-house
(PP-YOLOE+_s architecture), resources/onnx/chart_finder.onnx, with the
detection postprocess embedded in the ONNX graph.  It works on a
rasterised page, which lets it propose regions the layout GNN cannot reach: the
GNN classifies boxes derived from page primitives, whereas a chart drawn as a
few thousand unconnected vector paths has no single primitive box to classify.

Raw model boxes are then snapped to the page content.  Near duplicates are
merged, charts sitting side by side are split along their shared edge, and each
box grows to cover page primitives it only partially clips - axis labels,
legend text, small vector marks, embedded bitmaps.

`find_charts()` accepts the bundled ``fp32``, ``weight-fp16`` and ``full-fp16``
variants. Sessions are cached per model, provider list and device for the
lifetime of the process. All variants use the same page rendering, score
filtering and region refinement path.
"""

import threading
from pathlib import Path

import numpy as np
import pymupdf

from .common_util import compute_iou as _iou
from .onnx.common_util import make_session

__version__ = '260812'

MODEL_PATH = Path(__file__).resolve().parent / 'resources' / 'onnx' / 'chart_finder.onnx'
MODEL_VARIANTS = {
    'fp32': MODEL_PATH,
    'weight-fp16': MODEL_PATH.with_name('chart_finder_weight_fp16.onnx'),
    'full-fp16': MODEL_PATH.with_name('chart_finder_full_fp16.onnx'),
}

INPUT_SIZE = 640            # the model input is fixed at 640x640
RENDER_LONG_SIDE = 1024     # page raster long side, before the resize to INPUT_SIZE
MAX_DETECTIONS = 100
CHART_CLASS_ID = 0

# Refinement thresholds.  These are tuned against the detector's own box
# statistics and are not exposed as parameters, because moving one in isolation
# invalidates the rest.
_CONTAIN_OVERLAP = 0.90         # min-area overlap that reads as "one contains the other"
_DUPLICATE_IOU = 0.55
_PARTIAL_OVERLAP_MIN = 0.05
_PEER_AXIS_OVERLAP = 0.50       # row/column alignment required before splitting a pair
_MIN_RETAINED_AREA = 0.50       # a split may not shrink a box below this fraction
_CONFLICT_OVERLAP = 0.15        # of a contained pair, keep the smaller box only when
_CONFLICT_MARGIN = 0.05         # the larger one also conflicts with a third box
_TEXT_MAX_AREA_RATIO = 1.35
_TEXT_MAX_EXPAND = 48.0
_VECTOR_MAX_AREA_FRACTION = 0.25
_VECTOR_MAX_AREA_RATIO = 1.35
_VECTOR_MAX_EXPAND = 48.0
_RASTER_MAX_AREA_RATIO = 1.60
_RASTER_MAX_EXPAND = 72.0

_PROVIDER_ALIASES = {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'gpu': 'CUDAExecutionProvider',
    'tensorrt': 'TensorrtExecutionProvider',
}
_DEVICE_PROVIDERS = ('CUDAExecutionProvider', 'TensorrtExecutionProvider')

_sessions = {}
_sessions_lock = threading.Lock()


def find_charts(page, *, threshold=0.5, providers=None, device_id=0,
                model_path=None, variant=None):
    """
    Detect chart regions on one page.

    Parameters
    ----------
    page       : the pymupdf.Page to search.  It is never modified; a rotated
                 page is derotated on an in-memory copy.
    threshold  : minimum detection score, in [0, 1].
    providers  : ONNX Runtime execution providers.  None selects CPU.  Accepts
                 'cpu', 'cuda' or 'tensorrt', a comma-separated string of those,
                 or a sequence of aliases and/or full provider names.  Providers
                 the runtime does not offer are dropped, falling back to CPU.
    device_id  : GPU ordinal, used by the CUDA and TensorRT providers.
    model_path : an alternative chart_finder.onnx. Mutually exclusive with
                 ``variant``.
    variant    : bundled model variant: ``fp32`` (the default),
                 ``weight-fp16`` or ``full-fp16``.

    Returns
    -------
    list of dict
        One entry per chart, {'bbox': (x0, y0, x1, y1), 'score': float}.
        Coordinates are page points in the coordinate space of the derotated
        page, so for a rotated page they describe the visually upright page.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError('threshold must be in [0, 1]')
    if device_id < 0:
        raise ValueError('device_id must be non-negative')

    model_path = _resolve_model_path(model_path=model_path, variant=variant)
    session = _session(model_path, providers, device_id)
    page, owner = _upright(page)
    try:
        boxes, scores = _detect(page, session, float(threshold))
        boxes, scores = _refine(page, boxes, scores)
        return [{'bbox': tuple(box), 'score': score}
                for box, score in zip(boxes, scores)]
    finally:
        if owner is not None:
            owner.close()


# ---------------------------------------------------------------- runtime ----

def model_path_for_variant(variant='fp32'):
    """Return the bundled ONNX path for a named precision variant."""
    try:
        path = MODEL_VARIANTS[str(variant)]
    except KeyError as exc:
        raise ValueError(
            f'unknown chart finder variant {variant!r}; '
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


def _session(model_path, providers, device_id):
    """Return a cached InferenceSession for this model/provider/device."""
    path = str(Path(model_path).expanduser().resolve() if model_path else MODEL_PATH)
    names = _provider_names(providers)
    key = (path, names, device_id)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _build_session(path, names, device_id)
            _sessions[key] = session
    return session


def _provider_names(providers):
    """Normalise the `providers` argument to a tuple of ONNX provider names."""
    if providers is None:
        return ('CPUExecutionProvider',)
    if isinstance(providers, str):
        providers = providers.split(',')
    names = []
    for provider in providers:
        text = str(provider).strip()
        if not text:
            raise ValueError('providers contains an empty entry')
        names.append(_PROVIDER_ALIASES.get(text.lower(), text))
    return tuple(names)


def _build_session(path, names, device_id):
    """Create the session, dropping providers this runtime cannot offer."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    usable = [name for name in names if name in available]
    if not usable and 'CPUExecutionProvider' in available:
        usable = ['CPUExecutionProvider']
    if not usable:
        raise RuntimeError(f'no usable ONNX Runtime provider; requested={names},'
                           f' available={sorted(available)}')
    session = make_session(path, providers=[
        (name, {'device_id': device_id}) if name in _DEVICE_PROVIDERS else name
        for name in usable])
    inputs = {item.name: item.shape for item in session.get_inputs()}
    if set(inputs) != {'image', 'scale_factor'}:
        raise RuntimeError(f'unexpected chart finder inputs: {sorted(inputs)}')
    if list(inputs['image'][1:]) != [3, INPUT_SIZE, INPUT_SIZE]:
        raise RuntimeError(f'chart finder input shape {inputs["image"]} is not'
                           f' 3x{INPUT_SIZE}x{INPUT_SIZE}')
    return session


def _upright(page):
    """
    Return (page, owner) with the page rotation removed.

    A rotated page is copied into a new in-memory document and derotated there,
    leaving the caller's document untouched and writing nothing to disk.
    `owner` is the document to close, or None when the page was already upright.
    """
    if not page.rotation:
        return page, None
    owner = pymupdf.open()
    owner.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    upright = owner[0]
    upright.remove_rotation()
    return upright, owner


# ------------------------------------------------------------- inference ----

def _detect(page, session, threshold):
    """Run the model over the rendered page and return page-space boxes."""
    tensor, scale_factor, render_scale = _render(page)
    try:
        rows, counts = session.run(None, {'image': tensor,
                                          'scale_factor': scale_factor})
    except Exception as exc:
        # The exported NMS subgraph gathers from an empty tensor when the model
        # detects nothing at all.  Read that as "no charts on this page".
        message = str(exc)
        if 'Gather' not in message or 'axis 0 is not in valid range' not in message:
            raise
        rows = np.empty((0, 6), dtype=np.float32)
        counts = np.array([0], dtype=np.int32)

    count = min(int(np.asarray(counts).reshape(-1)[0]), len(rows))
    boxes = []
    scores = []
    for row in sorted(rows[:count].tolist(), key=lambda item: float(item[1]),
                      reverse=True):
        if len(boxes) >= MAX_DETECTIONS:
            break
        if int(row[0]) != CHART_CLASS_ID or float(row[1]) < threshold:
            continue
        box = _page_box(row[2:6], render_scale, page.rect)
        if box:
            boxes.append(box)
            scores.append(round(float(row[1]), 6))
    return boxes, scores


def _render(page):
    """
    Rasterise the page and return (tensor, scale_factor, render_scale).

    The page is rendered with its long side at RENDER_LONG_SIDE and then
    squashed - without preserving the aspect ratio - into the model's square
    input, which is how the model was trained.  `scale_factor` tells the
    in-graph postprocess how to map predictions back to raster pixels, and
    `render_scale` maps raster pixels back to page points.
    """
    rect = page.rect
    render_scale = float(RENDER_LONG_SIDE) / max(rect.width, rect.height)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(render_scale, render_scale),
                             alpha=False, colorspace=pymupdf.csRGB)
    raster = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n)[:, :, :3]
    resized = _resize_cubic(raster, INPUT_SIZE, INPUT_SIZE)
    tensor = resized.astype(np.float32)
    tensor /= 255.0
    tensor = tensor.transpose(2, 0, 1)[None]
    scale_factor = np.array([[float(INPUT_SIZE) / pixmap.height,
                              float(INPUT_SIZE) / pixmap.width]],
                            dtype=np.float32)
    return np.ascontiguousarray(tensor), scale_factor, render_scale


def _cubic_taps(src_len, dst_len):
    """Return the four source indices and weights per output position."""
    a = -0.75  # the cubic parameter OpenCV uses
    centres = (np.arange(dst_len, dtype=np.float64) + 0.5) * (src_len / dst_len) - 0.5
    base = np.floor(centres).astype(np.int64)
    t = (centres - base)[:, None]
    first = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
    second = ((a + 2) * t - (a + 3)) * t * t + 1
    third = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
    weights = np.concatenate([first, second, third,
                              1.0 - first - second - third], axis=1)
    taps = np.clip(base[:, None] + np.arange(-1, 3)[None, :], 0, src_len - 1)
    return taps, weights.astype(np.float32)


def _resize_cubic(image, width, height):
    """
    Bicubic resize matching cv2.resize(..., INTER_CUBIC) to within one level.

    The detector is sensitive to the resampling kernel: bilinear resizing moves
    boxes far enough to change which regions are reported, so the cubic kernel
    is reproduced here rather than substituted.  Doing it in numpy keeps the
    dependency list as it is - opencv is not required anywhere else.
    """
    src_height, src_width, channels = image.shape
    x_taps, x_weights = _cubic_taps(src_width, width)
    y_taps, y_weights = _cubic_taps(src_height, height)
    rows = image.astype(np.float32)
    # Vertical first: it shrinks the row count before the costlier column pass.
    vertical = np.zeros((height, src_width, channels), dtype=np.float32)
    for tap in range(4):
        vertical += rows[y_taps[:, tap], :, :] * y_weights[:, tap, None, None]
    out = np.zeros((height, width, channels), dtype=np.float32)
    for tap in range(4):
        out += vertical[:, x_taps[:, tap], :] * x_weights[:, tap, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _page_box(box, render_scale, rect):
    """Map one raster-pixel box to page points, clipped to the page."""
    x0, y0, x1, y1 = (float(value) / render_scale for value in box)
    x0 = max(rect.x0, min(rect.x1, x0))
    y0 = max(rect.y0, min(rect.y1, y0))
    x1 = max(rect.x0, min(rect.x1, x1))
    y1 = max(rect.y0, min(rect.y1, y1))
    if x1 <= x0 or y1 <= y0:
        return []
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


# ------------------------------------------------------------- refinement ----

def _refine(page, boxes, scores):
    """Merge, split and grow raw detections against the page content."""
    if not boxes:
        return [], []
    output, scores = _deduplicate(boxes, scores)
    output = _absorb(output, _text_boxes(page),
                     max_area_ratio=_TEXT_MAX_AREA_RATIO,
                     max_expand=_TEXT_MAX_EXPAND)
    output = _absorb(output, _vector_boxes(page),
                     max_area_ratio=_VECTOR_MAX_AREA_RATIO,
                     max_expand=_VECTOR_MAX_EXPAND,
                     candidate_area_fraction=_VECTOR_MAX_AREA_FRACTION)
    output = _absorb(output, _raster_boxes(page),
                     max_area_ratio=_RASTER_MAX_AREA_RATIO,
                     max_expand=_RASTER_MAX_EXPAND)
    output = _cut_overlaps(output)
    return [[round(float(value), 4) for value in box] for box in output], scores


def _deduplicate(boxes, scores):
    """
    Collapse detections that describe the same chart.

    Repeatedly take the most strongly overlapping pair and keep one box of it.
    The larger box wins, unless it also overlaps a third box appreciably more
    than the smaller one does, which marks it as a region spanning two charts.
    """
    current = [list(box) for box in boxes]
    current_scores = list(scores)
    while True:
        match = None
        best = 0.0
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                relation = max(
                    _minimum_overlap(current[left], current[right]) / _CONTAIN_OVERLAP,
                    _iou(current[left], current[right]) / _DUPLICATE_IOU)
                if relation >= 1.0 and relation > best:
                    match = (left, right)
                    best = relation
        if match is None:
            break
        left, right = match
        small, large = ((left, right)
                        if _area(current[left]) <= _area(current[right])
                        else (right, left))
        small_conflict = _other_conflict(current[small], current, {left, right})
        large_conflict = _other_conflict(current[large], current, {left, right})
        chosen = (small
                  if large_conflict >= _CONFLICT_OVERLAP
                  and small_conflict + _CONFLICT_MARGIN < large_conflict
                  else large)
        kept = list(current[chosen])
        kept_score = max(current_scores[left], current_scores[right])
        for index in sorted((left, right), reverse=True):
            del current[index]
            del current_scores[index]
        current.append(kept)
        current_scores.append(kept_score)
    return _cut_overlaps(current), current_scores


def _cut_overlaps(boxes):
    """
    Split pairs that overlap partially but are aligned as neighbours.

    Two charts in the same row - or the same column - whose boxes bleed into
    each other are cut apart along the middle of their intersection.  A cut is
    skipped when it would take either box below _MIN_RETAINED_AREA.
    """
    output = [list(box) for box in boxes]
    for left_index in range(len(output)):
        for right_index in range(left_index + 1, len(output)):
            left_box = output[left_index]
            right_box = output[right_index]
            intersection = _intersection_box(left_box, right_box)
            overlap = _minimum_overlap(left_box, right_box)
            if (intersection is None
                    or overlap < _PARTIAL_OVERLAP_MIN
                    or overlap >= _CONTAIN_OVERLAP):
                continue
            # axis 0 cuts a left/right pair, axis 1 cuts a top/bottom pair; the
            # pair must line up on the other axis to count as neighbours.
            split = None
            for axis in (0, 1):
                other = 1 - axis
                extent = min(left_box[other + 2] - left_box[other],
                             right_box[other + 2] - right_box[other])
                peers = _interval_overlap(
                    left_box[other], left_box[other + 2],
                    right_box[other], right_box[other + 2]) / max(extent, 1.0)
                if peers < _PEER_AXIS_OVERLAP:
                    continue
                first, second = ((left_index, right_index)
                                 if left_box[axis] + left_box[axis + 2]
                                 <= right_box[axis] + right_box[axis + 2]
                                 else (right_index, left_index))
                boundary = (intersection[axis] + intersection[axis + 2]) / 2.0
                near, far = list(output[first]), list(output[second])
                near[axis + 2] = boundary
                far[axis] = boundary
                split = (first, second, near, far)
                break
            if split is None:
                continue
            first, second, near, far = split
            if (_area(near) < _area(output[first]) * _MIN_RETAINED_AREA
                    or _area(far) < _area(output[second]) * _MIN_RETAINED_AREA):
                continue
            output[first], output[second] = near, far
    return output


def _absorb(boxes, candidates, *, max_area_ratio, max_expand,
            candidate_area_fraction=None):
    """
    Grow each box over the page primitives it only partially covers.

    A primitive the detector cut through - half an axis label, a legend swatch -
    is given to the box covering most of it, which then expands to include it.
    Fully covered primitives change nothing, and a box stops growing once it
    exceeds `max_area_ratio` of its area or `max_expand` points on a side.
    """
    output = [list(box) for box in boxes]
    anchors = [list(box) for box in boxes]
    for candidate in candidates:
        owners = []
        for index, box in enumerate(output):
            intersection = _intersection_area(box, candidate)
            coverage = intersection / max(_area(candidate), 1.0)
            if intersection <= 0 or coverage >= 0.999:
                continue
            if (candidate_area_fraction is not None
                    and _area(candidate) > _area(anchors[index]) * candidate_area_fraction):
                continue
            owners.append((coverage, intersection / max(_area(box), 1.0),
                           -index, index))
        if not owners:
            continue
        owner = max(owners)[-1]
        expanded = _union(output[owner], candidate)
        anchor = anchors[owner]
        if expanded == output[owner]:
            continue
        if (_area(expanded) > _area(anchor) * max_area_ratio
                or _max_side_expansion(anchor, expanded) > max_expand):
            continue
        output[owner] = expanded
    return output


# -------------------------------------------------------- page primitives ----

def _text_boxes(page):
    """Bounding boxes of all non-blank text spans."""
    boxes = []
    for block in page.get_text('dict').get('blocks', []):
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                if not str(span.get('text') or '').strip():
                    continue
                box = _candidate_box(span.get('bbox'))
                if box is not None:
                    boxes.append(box)
    return boxes


def _vector_boxes(page):
    """Bounding boxes of all vector drawings."""
    boxes = []
    for drawing in page.get_drawings():
        box = _candidate_box(drawing.get('rect'))
        if box is not None:
            boxes.append(box)
    return boxes


def _raster_boxes(page):
    """Bounding boxes of all displayed images, de-duplicated by position."""
    boxes = []
    seen = set()
    # xrefs=False: only the bbox is needed here, and resolving xrefs costs an
    # order of magnitude more per page without changing any bbox.
    for image in page.get_image_info(xrefs=False):
        box = _candidate_box(image.get('bbox'))
        if box is None or tuple(box) in seen:
            continue
        seen.add(tuple(box))
        boxes.append(box)
    return boxes


def _candidate_box(raw):
    """Coerce a bbox tuple or pymupdf.Rect to a list, or None if degenerate."""
    try:
        box = [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
    except (IndexError, TypeError, ValueError):
        try:
            box = [float(raw.x0), float(raw.y0), float(raw.x1), float(raw.y1)]
        except (AttributeError, TypeError, ValueError):
            return None
    return box if _area(box) > 0 else None


# --------------------------------------------------------------- geometry ----

def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_box(first, second):
    box = [max(first[0], second[0]), max(first[1], second[1]),
           min(first[2], second[2]), min(first[3], second[3])]
    return box if _area(box) > 0 else None


def _intersection_area(first, second):
    box = _intersection_box(first, second)
    return _area(box) if box is not None else 0.0


def _minimum_overlap(first, second):
    """Intersection as a fraction of the smaller box."""
    return _intersection_area(first, second) / max(
        min(_area(first), _area(second)), 1.0)


def _interval_overlap(first0, first1, second0, second1):
    return max(0.0, min(first1, second1) - max(first0, second0))


def _union(first, second):
    return [min(first[0], second[0]), min(first[1], second[1]),
            max(first[2], second[2]), max(first[3], second[3])]


def _max_side_expansion(anchor, expanded):
    """Largest outward move of any side, in points."""
    return max(anchor[0] - expanded[0], anchor[1] - expanded[1],
               expanded[2] - anchor[2], expanded[3] - anchor[3])


def _other_conflict(box, boxes, skip):
    """Strongest overlap of `box` with any box outside `skip`."""
    return max((_minimum_overlap(box, other) for index, other in enumerate(boxes)
                if index not in skip), default=0.0)
