"""
ImageFeatureExtractorV1
Purpose: ONNX-based image feature extraction and segmentation-based bbox detection.

CCL execution strategy (lazy):
    predict()               ? ONNX inference only. CCL is NOT run.
    is_image_page(page_img) ? runs predict() + non-picture CCL only.
    get_picture_detections()? runs picture CCL on first call after predict().

This minimises redundant work across three usage patterns:

  Pattern A  is_image_page() -> False (common case)
             ONNX: 1,  CCL: 1 (non-picture only)

  Pattern B  is_image_page() -> True -> predict() -> get_picture_detections()
             ONNX: 1,  CCL: 2 (non-picture + picture, each once)

  Pattern C  predict() -> get_feature_map() only
             ONNX: 1,  CCL: 0

Cache protocol (single-use):
    mark_cached() / consume_cache() ? let BoxRFDGNN.is_image_page() signal
    that ONNX inference has already run, so the next
    image_feature_extraction_task() call skips predict().
"""

import numpy as np

from ..common_util import resize_image, to_gray, extract_bboxes_from_segmentation, extract_bboxes_from_segmentation_numpy


# Class names expected from the segmentation head (index-aligned with channel axis)
_CLASS_NAMES = [
    'background', 'text', 'title', 'picture', 'table',
    'list-item', 'page-header', 'page-footer',
    'section-header', 'footnote', 'caption', 'formula',
]

_BACKGROUND_CLASS    = 'background'
_PICTURE_CLASS       = 'picture'
_NON_PICTURE_CLASSES = [c for c in _CLASS_NAMES
                        if c not in (_BACKGROUND_CLASS, _PICTURE_CLASS)]

_DETECTION_SCORE_THRESHOLD = 0.5
_MIN_COMPONENT_AREA        = 10
_MORPHOLOGY_KERNEL_SIZE    = 3


class ImageFeatureExtractorV1:
    """
    Wraps an ONNX inference session to produce:
      - a feature map
      - picture-class bbox detections      (lazy, via get_picture_detections())
      - non-picture-class detections       (lazy, via is_image_page())

    All CCL work is deferred until the result is actually needed, and cached
    so repeated calls within the same predict() cycle are free.
    """

    def __init__(self, onnx_session):
        """
        Args:
            onnx_session: onnxruntime.InferenceSession (or any object with
                          get_inputs() / run() matching the ORT interface)
        """
        self._session     = onnx_session
        self._feature_map = None   # (1, C, H, W) float32 ? raw ONNX output
        self._raw_outputs = None   # same as _feature_map; kept as alias for CCL input
        self._cached      = False  # cache flag for image_feature_extraction_task()

        # Lazy CCL results ? None means "not yet computed for this predict() cycle"
        self._detections_picture     = None  # list[dict] | None
        self._detections_non_picture = None  # list[dict] | None

        # Page / model resolution ? set during predict()
        self._page_h   = 0
        self._page_w   = 0
        self._target_h = 0
        self._target_w = 0

    # ------------------------------------------------------------------
    # Cache protocol (BoxRFDGNN + image_feature_extraction_task)
    # ------------------------------------------------------------------

    def mark_cached(self):
        """Signal that predict() results should be reused once."""
        self._cached = True

    def consume_cache(self):
        """
        Return True (and clear the flag) if a cached result exists.
        Called by image_feature_extraction_task() to skip predict().
        """
        if self._cached:
            self._cached = False
            return True
        return False

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, page_img, aug_fetmap=None):
        """
        Run ONNX inference and store the raw output.
        CCL is NOT performed here ? results are computed lazily on demand.

        Calling predict() invalidates all previously cached CCL results so
        that get_picture_detections() and is_image_page() always reflect the
        current page.

        Args:
            page_img:   np.ndarray (H, W, C), uint8
            aug_fetmap: optional extra channel map concatenated before inference

        Side effects:
            self._feature_map / self._raw_outputs <- (1, C, H, W)
            self._detections_picture     <- reset to None
            self._detections_non_picture <- reset to None
        """
        self._page_h, self._page_w = page_img.shape[:2]

        input_shape = self._session.get_inputs()[0].shape
        self._target_h, self._target_w = input_shape[2], input_shape[3]

        # Preprocess
        img_resized = resize_image(page_img, (self._target_w, self._target_h))
        img_gray    = to_gray(img_resized).astype(np.float32)

        min_val, max_val = img_gray.min(), img_gray.max()
        if max_val > min_val:
            img_gray = (img_gray - min_val) / (max_val - min_val)
        else:
            img_gray = np.zeros_like(img_gray, dtype=np.float32)

        nn_input = np.expand_dims(img_gray, axis=0)    # (1, H, W)
        if aug_fetmap is not None:
            nn_input = np.concatenate([nn_input, aug_fetmap], axis=0)
        nn_input = np.expand_dims(nn_input, axis=0)    # (1, C_in, H, W)

        # ONNX inference
        input_name   = self._session.get_inputs()[0].name
        ort_outputs  = self._session.run(None, {input_name: nn_input})[0]

        self._feature_map = ort_outputs
        self._raw_outputs = ort_outputs

        # Invalidate stale CCL results from the previous predict() cycle
        self._detections_picture     = None
        self._detections_non_picture = None

    # ------------------------------------------------------------------
    # Lazy CCL helpers
    # ------------------------------------------------------------------

    def _ensure_picture_detections(self):
        """Run picture CCL if not yet computed for the current predict() cycle."""
        if self._detections_picture is not None:
            return
        self._detections_picture = extract_bboxes_from_segmentation_numpy(
            self._raw_outputs,
            class_names=_CLASS_NAMES,
            target_class=[_PICTURE_CLASS],
            min_component_area=_MIN_COMPONENT_AREA,
        )

    def _ensure_non_picture_detections(self):
        """Run non-picture CCL if not yet computed for the current predict() cycle."""
        if self._detections_non_picture is not None:
            return
        self._detections_non_picture = extract_bboxes_from_segmentation_numpy(
            self._raw_outputs,
            class_names=_CLASS_NAMES,
            target_class=_NON_PICTURE_CLASSES,
            min_component_area=_MIN_COMPONENT_AREA,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_feature_map(self):
        """
        Return the raw feature map from the last predict() call.

        Returns:
            np.ndarray of shape (1, C, H, W), or None if predict() not yet called.
        """
        return self._feature_map

    def is_image_page(self, page):
        """
        Determine whether the page is an image-only PDF page that requires OCR.

        Returns True only when ALL three conditions hold:
          1. The page contains at least one raster image (pymupdf).
          2. The page contains no embedded selectable text (pymupdf).
          3. The segmentation model detects at least one non-picture region
             (text, table, header, etc.) inside the page image.

        Condition 1 + 2 confirm the PDF structure is image-only.
        Condition 3 confirms there is recoverable content inside the image.
        All three must hold for OCR to be both necessary and worthwhile.

        Calls predict() internally and marks the result as cached so the
        immediately following BoxRFDGNN.predict() skips re-running inference.
        picture CCL is deferred to get_picture_detections() if needed.

        Args:
            page: PyMuPDF page object

        Returns:
            bool
        """
        # Condition 1: page must contain at least one raster image
        if not page.get_image_info():
            return False

        # Condition 2: page must have no embedded selectable text
        if page.get_text("text").strip():
            return False

        # Run ONNX inference on the page pixmap and cache the result
        import numpy as np
        pix        = page.get_pixmap()
        bytes_data = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img   = bytes_data.reshape(pix.height, pix.width, pix.n)

        self.predict(page_img)
        self.mark_cached()

        # Condition 3: segmentation model must detect non-picture content
        self._ensure_non_picture_detections()

        return any(
            d['score'] > _DETECTION_SCORE_THRESHOLD
            for d in self._detections_non_picture
        )

    def get_picture_detections(self):
        """
        Return picture-class detections from the last predict() call,
        rescaled from model resolution to page pixel coordinates.

        Runs picture CCL on the first call after each predict() (lazy).
        Subsequent calls within the same predict() cycle are free.

        Returns:
            list of [x1, y1, x2, y2] in page pixel space.

        Raises:
            RuntimeError if predict() has not been called.
        """
        if self._raw_outputs is None:
            raise RuntimeError("predict() must be called before get_picture_detections()")

        self._ensure_picture_detections()

        resize_x = self._page_w / self._target_w
        resize_y = self._page_h / self._target_h

        bboxes = []
        for det in self._detections_picture:
            if det['score'] > _DETECTION_SCORE_THRESHOLD:
                b = det['bbox']
                bboxes.append([
                    b[0] * resize_x,
                    b[1] * resize_y,
                    b[2] * resize_x,
                    b[3] * resize_y,
                ])
        return bboxes
