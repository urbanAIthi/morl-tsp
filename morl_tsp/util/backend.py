# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""TraCI backend selection utilities.

This isolates backend resolution from ``env.py`` so the fallback behavior can
be tested without importing the full SUMO environment stack.
"""

from __future__ import annotations

import os
import warnings
from typing import Any


def _env_flag_true(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _install_libtraci_connection_shim(traci_backend: Any) -> None:
    if hasattr(traci_backend, "getConnection"):
        return

    def _switch_connection(label: str) -> None:
        """Switch the active libtraci connection before each proxied access."""
        traci_backend.switch(label)

    class _LibtraciConnectionProxy:
        """Proxy that makes libtraci behave like traci named connections."""

        def __init__(self, label: str):
            object.__setattr__(self, "_label", label)

        def __getattr__(self, name: str) -> Any:
            _switch_connection(object.__getattribute__(self, "_label"))
            return getattr(traci_backend, name)

        def close(self) -> None:
            _switch_connection(object.__getattribute__(self, "_label"))
            traci_backend.close()

    traci_backend.getConnection = lambda label: _LibtraciConnectionProxy(label)


def _resolve_traci_backend() -> tuple[Any, bool]:
    backend_raw = os.environ.get("MORL_TSP_TRACI_BACKEND")
    if backend_raw is None:
        prefer_libsumo = _env_flag_true(os.environ.get("MORL_TSP_USE_LIBSUMO"), default=True)
        backend = "libsumo" if prefer_libsumo else "libtraci"
    else:
        backend = str(backend_raw).strip().lower()

    if backend == "libsumo":
        try:
            import libsumo as traci_backend
        except Exception as exc:
            warnings.warn(
                f"Failed to import libsumo backend ({exc}). Falling back to traci backend.",
                stacklevel=2,
            )
            import traci as traci_backend  # type: ignore

            return traci_backend, False
        return traci_backend, True

    if backend == "libtraci":
        try:
            import libtraci as traci_backend
        except Exception as exc:
            warnings.warn(
                f"Failed to import libtraci backend ({exc}). Falling back to libsumo/traci backend.",
                stacklevel=2,
            )
            try:
                import libsumo as traci_backend
                return traci_backend, True
            except Exception:
                import traci as traci_backend  # type: ignore
                return traci_backend, False

        _install_libtraci_connection_shim(traci_backend)
        return traci_backend, False

    if backend == "traci":
        import traci as traci_backend  # type: ignore

        return traci_backend, False

    raise ValueError(
        "Invalid MORL_TSP_TRACI_BACKEND value. Expected one of: libsumo | libtraci | traci."
    )


traci, LIBSUMO = _resolve_traci_backend()
TRACI_SUPPORTS_CONNECTION_LABELS = hasattr(traci, "getConnection")

