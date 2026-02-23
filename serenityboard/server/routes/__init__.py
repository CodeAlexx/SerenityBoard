"""API route registration and watcher dependency injection."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serenityboard.server.run_manager import RunWatcher

__all__ = [
    "get_watcher",
    "set_watcher",
    "scalars_router",
    "images_router",
    "text_router",
    "hparams_router",
    "histograms_router",
    "distributions_router",
    "plugins_router",
    "traces_router",
    "eval_router",
    "artifacts_router",
    "metrics_router",
    "custom_scalars_router",
    "pr_curves_router",
    "audio_router",
    "graphs_router",
    "meshes_router",
    "embeddings_router",
    "notes_router",
]

_watcher: RunWatcher | None = None


def get_watcher():
    """FastAPI dependency that returns the active RunWatcher."""
    return _watcher


def set_watcher(w):
    """Set the global RunWatcher instance (called by app factory)."""
    global _watcher
    _watcher = w


# Import routers after get_watcher is defined (routes import get_watcher).
from serenityboard.server.routes.artifacts import router as artifacts_router  # noqa: E402
from serenityboard.server.routes.eval import router as eval_router  # noqa: E402
from serenityboard.server.routes.hparams import router as hparams_router  # noqa: E402
from serenityboard.server.routes.histograms import router as histograms_router  # noqa: E402
from serenityboard.server.routes.distributions import router as distributions_router  # noqa: E402
from serenityboard.server.routes.images import router as images_router  # noqa: E402
from serenityboard.server.routes.metrics import router as metrics_router  # noqa: E402
from serenityboard.server.routes.plugins import router as plugins_router  # noqa: E402
from serenityboard.server.routes.scalars import router as scalars_router  # noqa: E402
from serenityboard.server.routes.text import router as text_router  # noqa: E402
from serenityboard.server.routes.traces import router as traces_router  # noqa: E402
from serenityboard.server.routes.custom_scalars import router as custom_scalars_router  # noqa: E402
from serenityboard.server.routes.pr_curves import router as pr_curves_router  # noqa: E402
from serenityboard.server.routes.audio import router as audio_router  # noqa: E402
from serenityboard.server.routes.graphs import router as graphs_router  # noqa: E402
from serenityboard.server.routes.meshes import router as meshes_router  # noqa: E402
from serenityboard.server.routes.embeddings import router as embeddings_router  # noqa: E402
from serenityboard.server.routes.notes import router as notes_router  # noqa: E402
