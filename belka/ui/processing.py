"""Background developing: decoding, inversion, adjustments and thumbnails off the GUI thread.

The work is staged and cached so that dragging a slider redoes as little as
possible:

    decoded frame (per file stamp) ─► film analysis (per analysis key)
    ─► warped picture and its crop (per geometry)
    ─► inverted, cropped preview (per everything but the adjustments)
    ─► adjustments (every time)
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from belka.core import develop as dv
from belka.core import pipeline as pl
from belka.core.film import FilmProfile
from belka.core.rawio import LinearImage
from belka.core.session import Frame, Session, load_frame

PREVIEW_SIDE = 2000
# A crop this much smaller than the screen it fills is developed from the
# raw's native half-size decode (3024 px on the D780) instead of the preview;
# a slightly tightened full frame is not worth four times the pixels.
DETAIL_MARGIN = 1.5
_tokens = itertools.count(1)


@dataclass
class DevelopJob:
    # "render", "thumb", "before", "sample_base", "neutral", "upright" or "auto_tone"
    kind: str
    session: Session
    frame: Frame
    settings: pl.DevelopSettings
    profile: FilmProfile
    view: str = "positive"  # or "negative"
    ignore_crop: bool = False
    rect: tuple[float, float, float, float] | None = None
    mode: str = ""  # upright mode
    max_side: int = 1600
    # Width/height the crop tool is locked to (None: free); sizes its bounds.
    crop_ratio: float | None = None
    fields: tuple[str, ...] | None = None  # auto tone: the sliders to set (None: all)
    token: int = field(default_factory=lambda: next(_tokens))


@dataclass
class DevelopResult:
    job: DevelopJob
    image: QImage | None = None
    histogram: np.ndarray | None = None
    analysis: pl.Analysis | None = None
    value: object = None
    error: str = ""
    meta: dict = field(default_factory=dict)
    rgb8: np.ndarray | None = None
    crop: tuple | None = None  # effective crop in the warped image (for the crop tool)
    valid: tuple | None = None  # largest valid rect of the warp (crop bounds)


def _stamp(path) -> tuple:
    try:
        st = path.stat()
        return (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(path), None, None)


def array_to_qimage(rgb8: np.ndarray) -> QImage:
    rgb8 = np.ascontiguousarray(rgb8)
    h, w = rgb8.shape[:2]
    return QImage(rgb8.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def histogram(rgb8: np.ndarray) -> np.ndarray:
    flat = rgb8.reshape(-1, 3)
    step = max(1, flat.shape[0] // 400_000)
    flat = flat[::step]
    return np.stack([np.bincount(flat[:, c], minlength=256) for c in range(3)])


class DevelopWorker(QObject):
    finished = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._images: OrderedDict[tuple, LinearImage] = OrderedDict()
        self._analyses: OrderedDict[tuple, pl.Analysis] = OrderedDict()
        self._warped: OrderedDict[tuple, tuple] = OrderedDict()
        self._inverted: OrderedDict[tuple, np.ndarray] = OrderedDict()

    @staticmethod
    def _remember(cache: OrderedDict, key, value, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _image_key(self, session: Session, frame: Frame) -> tuple:
        # File stamps, not just names: a recapture after deleting the last
        # frame reuses the same id and file name, and re-shooting a flat
        # rewrites the same .npy.
        files = [session.resolve(rel) for rel in frame.files]
        flats = [session.resolve(rel) for _key, rel in sorted(session.flats.items())]
        return (str(session.path), frame.id, tuple(_stamp(p) for p in files), tuple(_stamp(p) for p in flats))

    def _image(self, session: Session, frame: Frame, detail: bool = False) -> LinearImage:
        key = (self._image_key(session, frame), detail)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        image = load_frame(session, frame, half_size=True, max_side=None if detail else PREVIEW_SIDE)
        self._remember(self._images, key, image, 8)
        return image

    def _source(self, job: DevelopJob, preview: LinearImage, analysis: pl.Analysis) -> LinearImage:
        """The preview, or the sharper native decode when the crop would be shown enlarged."""
        if job.kind not in ("render", "before") or job.ignore_crop or job.view != "positive":
            return preview
        s = job.settings
        h, w = preview.rgb.shape[:2]
        if s.rotation % 180:
            w, h = h, w
        crop = dv.effective_crop(s, analysis, w, h)
        if crop is None:
            return preview
        shown = max((crop[2] - crop[0]) * w, (crop[3] - crop[1]) * h)
        if shown * DETAIL_MARGIN >= job.max_side:
            return preview
        detail = self._image(job.session, job.frame, detail=True)
        return detail if detail.rgb.shape[1] > preview.rgb.shape[1] else preview

    def _analysis(self, job: DevelopJob, image: LinearImage) -> pl.Analysis:
        # Always measured on the preview: the same base, levels and frame at any resolution.
        key = (self._image_key(job.session, job.frame), job.settings.analysis_key(), repr(job.profile))
        if key in self._analyses:
            return self._analyses[key]
        analysis = pl.analyze(image.rgb, job.settings, job.profile)
        self._remember(self._analyses, key, analysis, 32)
        return analysis

    def _geometry(self, job: DevelopJob, image: LinearImage, analysis: pl.Analysis) -> tuple:
        """Warped picture, the crop it gets and the crop tool's bounds.

        Lens correction and the warp take ~70 ms on a preview; dragging an
        adjustment slider must not redo them.
        """
        s = job.settings
        key = (self._image_key(job.session, job.frame), image.rgb.shape, s.geometry_key(), s.analysis_key(),
               s.crop_aspect, s.constrain_crop, repr(job.profile), job.crop_ratio)
        if key in self._warped:
            self._warped.move_to_end(key)
            return self._warped[key]
        warped = dv.geometry(image.rgb, s, analysis, apply_crop=False, profile=job.profile)
        h, w = warped.shape[:2]
        crop = dv.effective_crop(s, analysis, w, h)
        valid = None
        if not dv.geometry_is_identity(s):
            from belka.core import transform

            valid = tuple(transform.largest_valid_rect(s, w, h, aspect=job.crop_ratio))
        entry = (warped, crop, valid)
        self._remember(self._warped, key, entry, 2)
        return entry

    @Slot(object)
    def run(self, job: DevelopJob) -> None:
        try:
            result = self._run(job)
        except Exception as exc:  # report, never kill the worker thread
            result = DevelopResult(job=job, error=f"{type(exc).__name__}: {exc}")
        self.finished.emit(result)

    def _run(self, job: DevelopJob) -> DevelopResult:
        preview = self._image(job.session, job.frame)
        settings = job.settings
        # Tools work on the whole warped view, without the crop.
        view_settings = settings.copy(crop=None, auto_crop=False) if job.ignore_crop else settings
        rgb_seq = bool(preview.meta.get("rgb_sequential"))
        analysis = self._analysis(job, preview)

        if job.kind == "upright":
            return DevelopResult(job=job, value=self._upright(job, preview, analysis))
        image = self._source(job, preview, analysis)

        warped, real_crop, valid = self._geometry(job, image, analysis)
        if job.kind == "auto_tone":
            from belka.core import autotone

            geo = warped if real_crop is None else warped[pl.crop_slices(warped.shape, real_crop)]
            value = autotone.auto_tone(geo, analysis, settings, job.profile, image.camera_matrix, rgb_seq,
                                       fields=job.fields)
            return DevelopResult(job=job, value=value)
        if job.kind in ("sample_base", "neutral"):
            patch = warped[pl.crop_slices(warped.shape, job.rect)]
            if job.kind == "sample_base":
                value = tuple(float(v) for v in np.median(patch.reshape(-1, 3), axis=0))
                return DevelopResult(job=job, value=value)
            offsets = pl.neutral_offsets(patch, analysis, settings, job.profile, image.camera_matrix, rgb_seq)
            return DevelopResult(job=job, value=offsets, analysis=analysis)

        # The tools show the whole picture; the crop tool starts from the real crop.
        crop = None if job.ignore_crop else real_crop
        geo = warped if crop is None else warped[pl.crop_slices(warped.shape, crop)]
        if job.view == "negative":
            geo = pl.downsample(geo, job.max_side)
            peak = float(np.percentile(geo.max(axis=-1), 99.5)) or 1.0
            out = pl.srgb_encode(geo / peak)
        else:
            # The crop actually applied: the tools' uncropped view and the normal
            # cropped one can share every setting.
            key = (self._image_key(job.session, job.frame), image.rgb.shape, repr(dv.adjust_free(view_settings)),
                   job.max_side, crop, repr(job.profile))
            display = self._inverted.get(key)
            if display is None:
                small = pl.downsample(geo, job.max_side)
                display = dv.invert(small, analysis, view_settings, job.profile, image.camera_matrix, rgb_seq)
                self._remember(self._inverted, key, display, 6)
            # Pixel radii (sharpening, grain...) are defined at full resolution.
            full_w, full_h = image.meta.get("full_size") or (image.rgb.shape[1], image.rgb.shape[0])
            if settings.rotation % 180:
                full_w = full_h  # the displayed width is the sensor's height
            crop_w = 1.0 if crop is None else max(crop[2] - crop[0], 1e-3)
            scale = display.shape[1] / max(full_w * crop_w, 1.0)
            out = dv.finish(display, view_settings, scale, seed=dv.frame_seed(job.frame.id))
            if settings.output == "flat":
                out = pl.srgb_encode(out)
        rgb8 = pl.to_uint8(out)
        return DevelopResult(
            job=job,
            image=array_to_qimage(rgb8),
            histogram=histogram(rgb8) if job.kind == "render" else None,
            analysis=analysis,
            meta=dict(image.meta),
            rgb8=rgb8 if job.kind == "render" else None,
            crop=real_crop,
            valid=valid,
        )

    @staticmethod
    def _upright(job: DevelopJob, image: LinearImage, analysis: pl.Analysis) -> dict:
        """Lines are looked for inside the frame: film edges and sprocket holes must not vote.

        A manual crop is mapped back through the transform, so the region moves
        with the correction; iterating to a fixed point makes pressing the same
        Upright button twice give the same result.
        """
        from belka.core import transform

        settings = job.settings
        oriented = pl.downsample(pl.orient(image.rgb, settings.rotation, settings.flip_h, settings.flip_v),
                                 transform.UPRIGHT_SIDE)
        if settings.section_on("lens"):
            oriented = transform.lens_correct(oriented, settings, job.profile)
        h, w = oriented.shape[:2]

        def region_for(s: pl.DevelopSettings):
            if s.crop is None:
                return analysis.extra.get("frame") if s.auto_crop else None
            x0, y0, x1, y1 = s.crop
            pts = transform.map_points_to_source([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], s, w, h)
            lo, hi = np.clip(pts.min(axis=0), 0, 1), np.clip(pts.max(axis=0), 0, 1)
            return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

        current = settings
        result: dict = {}
        for _step in range(4):
            result = transform.auto_upright(oriented, job.mode, region=region_for(current))
            if not result or settings.crop is None:
                break
            following = settings.copy(persp_rotate=0.0, persp_aspect=0.0, **result)
            if all(abs(getattr(following, k) - getattr(current, k)) < 0.02 for k in result):
                break
            current = following
        return result


class DevelopController(QObject):
    """Feeds the worker one job at a time; newer renders replace stale ones."""

    rendered = Signal(object)
    before = Signal(object)
    thumbnail = Signal(object)
    sampled = Signal(object)
    failed = Signal(str)
    _submit = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.thread = QThread(self)
        self.thread.setObjectName("belka-develop")
        self.worker = DevelopWorker()
        self.worker.moveToThread(self.thread)
        self._submit.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished)
        self._busy = False
        self._pending_render: DevelopJob | None = None
        self._pending_before: DevelopJob | None = None
        self._queue: list[DevelopJob] = []
        self._thumbs: list[DevelopJob] = []
        self.thread.start()

    @staticmethod
    def _detach(job: DevelopJob) -> DevelopJob:
        # The worker must not see the GUI thread mutating the roll mid-job.
        session = replace(job.session, flats=dict(job.session.flats), frames=[])
        frame = replace(job.frame, files=list(job.frame.files), lights=[list(l) for l in job.frame.lights])
        return replace(job, session=session, frame=frame, settings=job.settings.copy())

    def render(self, job: DevelopJob) -> None:
        self._pending_render = self._detach(job)
        self._pump()

    def render_before(self, job: DevelopJob) -> None:
        self._pending_before = self._detach(replace(job, kind="before"))
        self._pump()

    def request(self, job: DevelopJob) -> None:
        self._queue.append(self._detach(job))
        self._pump()

    def thumbnail_for(self, job: DevelopJob) -> None:
        self._thumbs = [j for j in self._thumbs if j.frame.id != job.frame.id]
        self._thumbs.append(self._detach(job))
        self._pump()

    def cancel_thumbnails(self) -> None:
        self._thumbs.clear()

    def _pump(self) -> None:
        if self._busy:
            return
        job = None
        if self._queue:
            job = self._queue.pop(0)
        elif self._pending_render is not None:
            job, self._pending_render = self._pending_render, None
        elif self._pending_before is not None:
            job, self._pending_before = self._pending_before, None
        elif self._thumbs:
            job = self._thumbs.pop(0)
        if job is None:
            return
        self._busy = True
        self._submit.emit(job)

    def _on_finished(self, result: DevelopResult) -> None:
        self._busy = False
        if result.error and result.job.kind == "thumb":
            pass  # the filmstrip keeps its placeholder; the loupe reports the problem
        elif result.error:
            self.failed.emit(result.error)
        elif result.job.kind == "render":
            self.rendered.emit(result)
        elif result.job.kind == "before":
            self.before.emit(result)
        elif result.job.kind == "thumb":
            self.thumbnail.emit(result)
        else:
            self.sampled.emit(result)
        self._pump()

    def shutdown(self) -> None:
        self._queue.clear()
        self._thumbs.clear()
        self._pending_render = None
        self._pending_before = None
        self.thread.quit()
        self.thread.wait(10000)
