"""Process-runtime compatibility helpers for preprocessing.

The conda Python 3.12 build used by this project exposes an ``RLock`` without
CPython's private ``_recursion_count`` method.  multiprocess 0.70.18 calls that
private method from ``ResourceTracker.__del__``, producing a harmless but noisy
shutdown traceback even when all pools closed correctly.  Patch only that
version-dependent introspection while retaining fd closure and child waiting.
"""
from __future__ import annotations

import os


def configure_multiprocess_resource_tracker() -> None:
    """Install a narrow compatibility patch when the private RLock API is absent."""
    import threading
    import multiprocess.resource_tracker as resource_tracker

    probe = threading.RLock()
    if hasattr(probe, "_recursion_count"):
        return
    tracker_class = resource_tracker.ResourceTracker
    if getattr(tracker_class, "_plantgeneann_compat", False):
        return

    def _stop_locked(self, close=os.close, waitpid=os.waitpid, **_ignored):
        if self._fd is None or self._pid is None:
            return
        close(self._fd)
        self._fd = None
        try:
            waitpid(self._pid, 0)
        except ChildProcessError:
            pass
        self._pid = None

    tracker_class._stop_locked = _stop_locked
    tracker_class._plantgeneann_compat = True
