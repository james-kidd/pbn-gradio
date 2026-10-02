"""Shared helpers for the notebooks: environment bootstrap, demo scenes, display.

Deliberately empty of imports. `nbtools.bootstrap` is what installs the
third-party packages, and `scenes` and `display` need some of those at module
scope (`cv2`, `matplotlib`). Re-exporting them here would mean that

    from nbtools.bootstrap import setup

executed this file, which imported `scenes`, which imported `cv2` — and failed
on any environment that did not already have it, which is precisely the
environment `setup()` exists to fix.

So notebooks call `setup()` first and import `scenes` and `display` after.
"""
