"""Plugin base class and route decorator for SerenityBoard."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serenityboard.server.data_provider import RunDataProvider

__all__ = ["Plugin", "route"]


def route(method: str, path: str):
    """Decorator to mark a method as a plugin route."""
    def decorator(func):
        func._route_method = method
        func._route_path = path
        return func
    return decorator


class Plugin:
    """Base class for SerenityBoard plugins.

    Subclass this and set name + display_name. Register via entry points.
    """
    name: str = ""
    display_name: str = ""
    frontend_module: str | None = None

    def is_active(self, run_name: str, provider: RunDataProvider) -> bool:
        """Return True if this run has data for this plugin."""
        return False

    def get_routes(self) -> list[tuple[str, str, callable]]:
        """Discover all @route-decorated methods."""
        routes = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, '_route_method'):
                routes.append((attr._route_method, attr._route_path, attr))
        return routes
