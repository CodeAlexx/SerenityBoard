"""Compatibility shim for ``torch.utils.tensorboard.SummaryWriter``.

Users change one import line to switch from TensorBoard to SerenityBoard::

    # from torch.utils.tensorboard import SummaryWriter
    from serenityboard.compat.torch import SummaryWriter

All supported methods keep their original TensorBoard signatures.  Argument
names are translated automatically (``global_step`` -> ``step``, etc.).

``torch`` is **not** imported at module level -- it is only needed inside
``add_figure`` when the figure canvas produces a torch tensor, so every other
method works without PyTorch installed.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from serenityboard.writer.summary_writer import SummaryWriter as _NativeWriter

__all__ = ["SummaryWriter"]

# TensorBoard uses ``bins='tensorflow'`` as default for add_histogram.
# That corresponds to 30 bins in their code but our native writer defaults
# to 64 which gives a nicer chart.  We map the string to an integer.
_TB_BINS_MAP = {
    "tensorflow": 64,
    "auto": 64,
    "fd": 64,
    "doane": 64,
    "scott": 64,
    "stone": 64,
    "rice": 64,
    "sturges": 64,
    "sqrt": 64,
}

_V2_METHODS = (
    "add_video",
)


def _not_implemented(method_name: str) -> None:
    raise NotImplementedError(
        f"SummaryWriter.{method_name}() is not yet supported by SerenityBoard."
    )


class SummaryWriter:
    """Drop-in replacement for ``torch.utils.tensorboard.SummaryWriter``.

    Wraps :class:`serenityboard.SummaryWriter` and translates TensorBoard
    argument names to SerenityBoard equivalents.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        *,
        comment: str = "",
        purge_step: int | None = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
        **kwargs: Any,
    ) -> None:
        # Map TensorBoard constructor args to SerenityBoard equivalents.
        native_kwargs: dict[str, Any] = {}

        if log_dir is not None:
            native_kwargs["logdir"] = log_dir
        elif "logdir" in kwargs:
            native_kwargs["logdir"] = kwargs.pop("logdir")
        else:
            native_kwargs["logdir"] = "runs"

        if purge_step is not None:
            native_kwargs["resume_step"] = purge_step

        # Map queue/flush parameters
        if max_queue != 10:  # non-default
            native_kwargs["max_queue_size"] = max_queue
        if flush_secs != 120:  # non-default
            native_kwargs["flush_interval_secs"] = float(flush_secs)

        # Silently ignored TensorBoard args
        if comment:
            pass  # TensorBoard appends comment to logdir; we ignore it
        if filename_suffix:
            pass  # TensorBoard uses this for event file naming

        # Pass through any remaining kwargs the native writer accepts
        native_kwargs.update(kwargs)

        self._writer = _NativeWriter(**native_kwargs)

    # ── Supported methods (full) ──────────────────────────────────────

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
        double_precision: bool = False,
    ) -> None:
        """Log a scalar value.

        Maps ``scalar_value`` -> ``value`` and ``global_step`` -> ``step``.
        """
        step = global_step if global_step is not None else 0
        self._writer.add_scalar(tag, float(scalar_value), step=step)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Log multiple scalars under *main_tag*.

        Maps ``tag_scalar_dict`` -> ``values`` and ``global_step`` -> ``step``.
        """
        step = global_step if global_step is not None else 0
        self._writer.add_scalars(main_tag, values=tag_scalar_dict, step=step)

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int | None = None,
        walltime: float | None = None,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image.

        Maps ``img_tensor`` -> ``img`` and ``global_step`` -> ``step``.
        """
        step = global_step if global_step is not None else 0
        self._writer.add_image(tag, img=img_tensor, step=step, dataformats=dataformats)

    def add_images(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int | None = None,
        walltime: float | None = None,
        dataformats: str = "NCHW",
    ) -> None:
        """Log a batch of images.

        Maps ``img_tensor`` -> ``imgs`` and ``global_step`` -> ``step``.
        """
        step = global_step if global_step is not None else 0
        self._writer.add_images(tag, imgs=img_tensor, step=step, dataformats=dataformats)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: int | None = None,
        bins: str | int = "tensorflow",
        walltime: float | None = None,
        max_bins: int | None = None,
    ) -> None:
        """Log a histogram.

        ``bins='tensorflow'`` (and other string presets) maps to ``bins=64``.
        Integer values pass through directly.
        """
        step = global_step if global_step is not None else 0
        if isinstance(bins, str):
            resolved_bins = _TB_BINS_MAP.get(bins, 64)
        else:
            resolved_bins = int(bins)
        if max_bins is not None:
            resolved_bins = min(resolved_bins, max_bins)
        self._writer.add_histogram(tag, values=values, step=step, bins=resolved_bins)

    def add_distribution(
        self,
        tag: str,
        values: Any,
        global_step: int | None = None,
        bins: str | int = "tensorflow",
        walltime: float | None = None,
        max_bins: int | None = None,
    ) -> None:
        """Log a distribution (maps to histogram internally).

        This is a SerenityBoard extension that records the same underlying
        histogram data but can be rendered as a percentile band chart in the UI.
        """
        self.add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Log a text string.

        Maps ``text_string`` -> ``text`` and ``global_step`` -> ``step``.
        """
        step = global_step if global_step is not None else 0
        self._writer.add_text(tag, text=text_string, step=step)

    def add_hparams(
        self,
        hparam_dict: dict,
        metric_dict: dict,
        hparam_domain_discrete: dict | None = None,
        run_name: str | None = None,
    ) -> None:
        """Log hyperparameters and their metrics. Direct passthrough."""
        self._writer.add_hparams(hparam_dict, metric_dict)

    # ── Partial support ───────────────────────────────────────────────

    def add_figure(
        self,
        tag: str,
        figure: Any,
        global_step: int | None = None,
        close: bool = True,
        walltime: float | None = None,
    ) -> None:
        """Convert a matplotlib figure to an image and log it.

        Renders the figure to a numpy array via the Agg backend, then
        delegates to :meth:`add_image`.  If *close* is ``True`` (the
        default) the figure is closed after rendering.
        """
        try:
            canvas = figure.canvas
            canvas.draw()
            # Get RGBA buffer from the canvas
            buf = canvas.buffer_rgba()
            img_array = np.frombuffer(buf, dtype=np.uint8).reshape(
                canvas.get_width_height()[::-1] + (4,)
            )
            # Convert RGBA -> RGB
            img_array = img_array[:, :, :3]
        except Exception:
            # Fallback: use savefig to BytesIO
            import io

            from PIL import Image

            buf = io.BytesIO()
            figure.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            pil_img = Image.open(buf)
            img_array = np.array(pil_img.convert("RGB"))

        step = global_step if global_step is not None else 0
        # img_array is HWC format
        self._writer.add_image(tag, img=img_array, step=step, dataformats="HWC")

        if close:
            try:
                import matplotlib.pyplot as plt

                plt.close(figure)
            except ImportError:
                pass

    # ── Lifecycle ─────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all pending data to disk."""
        self._writer.flush()

    def close(self) -> None:
        """Flush and close the writer."""
        self._writer.close()

    def __enter__(self) -> SummaryWriter:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        if exc_type is not None:
            try:
                self.close()
            except Exception:
                pass  # Don't mask the original exception
        else:
            self.close()
        return False

    # ── V2 stubs ──────────────────────────────────────────────────────

    def add_video(self, *args: Any, **kwargs: Any) -> None:
        _not_implemented("add_video")

    def add_audio(
        self,
        tag: str,
        snd_tensor: Any,
        global_step: int | None = None,
        sample_rate: int = 44100,
        walltime: float | None = None,
    ) -> None:
        """Log an audio clip. Maps ``snd_tensor`` and ``global_step``."""
        step = global_step if global_step is not None else 0
        self._writer.add_audio(tag, audio_data=snd_tensor, step=step, sample_rate=sample_rate)

    def add_graph(
        self,
        model: Any,
        input_to_model: Any = None,
        verbose: bool = False,
        use_strict_trace: bool = True,
    ) -> None:
        """Log a PyTorch model computation graph.

        Maps TensorBoard's ``add_graph`` to the native writer.
        """
        self._writer.add_graph(
            model=model,
            input_to_model=input_to_model,
            verbose=verbose,
            use_strict_trace=use_strict_trace,
        )

    def add_embedding(
        self,
        mat: Any,
        metadata: Any = None,
        label_img: Any = None,
        global_step: int | None = None,
        tag: str = "default",
        metadata_header: Any = None,
    ) -> None:
        """Log an embedding matrix for projector visualization.

        Maps ``global_step`` -> ``global_step`` (defaulting to 0 when None).
        """
        self._writer.add_embedding(
            mat=mat,
            metadata=metadata,
            label_img=label_img,
            global_step=global_step if global_step is not None else 0,
            tag=tag,
            metadata_header=metadata_header,
        )

    def add_pr_curve(
        self,
        tag: str,
        labels: Any,
        predictions: Any,
        global_step: int | None = None,
        num_thresholds: int = 127,
        weights: Any = None,
        walltime: float | None = None,
    ) -> None:
        """Log a precision-recall curve. Maps TensorBoard args."""
        step = global_step if global_step is not None else 0
        self._writer.add_pr_curve(
            tag, labels=labels, predictions=predictions,
            step=step, num_thresholds=num_thresholds,
        )

    def add_pr_curve_raw(
        self,
        tag: str,
        true_positive_counts: Any,
        false_positive_counts: Any,
        true_negative_counts: Any,
        false_negative_counts: Any,
        precision: Any,
        recall: Any,
        global_step: int | None = None,
        num_thresholds: int = 127,
        weights: Any = None,
        walltime: float | None = None,
    ) -> None:
        """Log a pre-computed PR curve. Maps TensorBoard args."""
        step = global_step if global_step is not None else 0
        self._writer.add_pr_curve_raw(
            tag,
            true_positive_counts=true_positive_counts,
            false_positive_counts=false_positive_counts,
            true_negative_counts=true_negative_counts,
            false_negative_counts=false_negative_counts,
            precision=precision,
            recall=recall,
            step=step,
        )

    def add_mesh(
        self,
        tag: str,
        vertices: Any,
        colors: Any = None,
        faces: Any = None,
        config_dict: dict | None = None,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Log a 3D mesh with vertices, optional colors, and optional faces.

        Maps ``global_step`` (defaulting to 0 when None) and passes through
        ``walltime``.
        """
        self._writer.add_mesh(
            tag=tag,
            vertices=vertices,
            colors=colors,
            faces=faces,
            config_dict=config_dict,
            global_step=global_step if global_step is not None else 0,
            walltime=walltime,
        )

    def add_custom_scalars(self, layout: Any) -> None:
        """Store a custom scalars layout.

        Accepts TensorBoard's layout dict format and converts to SB format.
        TB format: nested dict with category -> chart -> (multiline, tag_regex_list)
        SB format: {"categories": [{"title": ..., "charts": [{"title": ..., "tags": [...]}]}]}
        """
        # Convert TB protobuf-style layout to SB JSON format
        sb_categories = []
        if isinstance(layout, dict):
            for category_name, charts in layout.items():
                sb_charts = []
                if isinstance(charts, dict):
                    for chart_name, chart_config in charts.items():
                        # TB format: chart_config is typically (Multiline|Margin, [tag_regex, ...])
                        tags = []
                        if isinstance(chart_config, (list, tuple)) and len(chart_config) >= 2:
                            tag_list = chart_config[1] if isinstance(chart_config[1], (list, tuple)) else [chart_config[1]]
                            tags = [str(t) for t in tag_list]
                        elif isinstance(chart_config, (list, tuple)):
                            tags = [str(t) for t in chart_config]
                        sb_charts.append({"title": chart_name, "tags": tags})
                sb_categories.append({"title": category_name, "charts": sb_charts})

        sb_layout = {"categories": sb_categories}
        self._writer.add_custom_scalars_layout(sb_layout)
