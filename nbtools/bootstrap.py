"""
One call that makes the toolkit importable, in Colab or in a terminal.

Colab opens a single `.ipynb` straight from GitHub: none of the repository is
on disk, the toolkit is not installed, and the runtime resets on every restart.
A terminal has the opposite problem — everything is already there and cloning
again would be wrong. `setup()` handles both by looking before it fetches.

It also works around one real coupling in the source repo. `pipeline.py`
imports Gradio at module scope, so importing *any* of the compute layer drags
in the whole web-UI dependency — about a hundred megabytes that a notebook has
no use for. `land_use.py` and `siting.py` already defer their Gradio import
into the function bodies that raise `gr.Error`; `pipeline.py` does not, and the
inconsistency is almost certainly an oversight rather than a decision.

Rather than install Gradio to satisfy one exception class, `setup()` registers
a six-line stub exposing `Error`. The stub is used *only* when Gradio is
genuinely absent, so a machine that has it keeps the real thing. See the README
for the one-line upstream fix that makes the stub unnecessary.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import types
from dataclasses import dataclass, field

# The toolkit currently lives on the branch of an open pull request. Point
# these at `main` once it merges — that is the only edit needed here.
SOURCE_REPO = "https://github.com/james-kidd/pbn-gradio.git"
SOURCE_BRANCH = "claude/area-labeling-potential-use-z9ex0w"

# Import-name → pip-name, for the few that differ.
REQUIRED = {
    "numpy": "numpy",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "matplotlib": "matplotlib",
}

TOOLKIT_MODULES = (
    "geo",
    "segmenters",
    "topology",
    "optimize",
    "pipeline",
    "land_use",
    "siting",
)


@dataclass
class Toolkit:
    """The imported modules, plus how they were found."""
    source_path: pathlib.Path
    cloned: bool
    gradio_stubbed: bool
    installed: list[str] = field(default_factory=list)
    modules: dict[str, types.ModuleType] = field(default_factory=dict)

    def __getattr__(self, name: str) -> types.ModuleType:
        # Lets callers write `tk.segmenters` instead of `tk.modules["segmenters"]`.
        try:
            return self.modules[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def summary(self) -> str:
        lines = [
            f"toolkit   {self.source_path}"
            + ("  (cloned)" if self.cloned else "  (found locally)"),
            f"modules   {', '.join(sorted(self.modules))}",
        ]
        if self.installed:
            lines.append(f"installed {', '.join(self.installed)}")
        if self.gradio_stubbed:
            lines.append("gradio    stubbed (UI layer not needed for these demos)")
        return "\n".join(lines)


# ── Locating things ───────────────────────────────────────────

def in_colab() -> bool:
    return "google.colab" in sys.modules or pathlib.Path("/content").is_dir()


def _workdir() -> pathlib.Path:
    """Where to clone into: Colab's writable root, or beside the notebook."""
    return pathlib.Path("/content") if in_colab() else pathlib.Path.cwd()


def find_source(start: pathlib.Path | None = None) -> pathlib.Path | None:
    """Look for an existing checkout of the toolkit before cloning one.

    Checks the working directory and its parents, then any sibling directory —
    the common layout when both repositories are cloned next to each other.
    """
    start = start or pathlib.Path.cwd()
    candidates = [start, *start.parents]
    for base in list(candidates):
        candidates.extend(p for p in base.glob("*") if p.is_dir())
    for base in candidates:
        if (base / "segmenters.py").exists() and (base / "pipeline.py").exists():
            return base.resolve()
    return None


def clone_source(dest: pathlib.Path | None = None, quiet: bool = True) -> pathlib.Path:
    """Shallow-clone the pinned branch of the toolkit."""
    dest = dest or (_workdir() / "pbn-gradio")
    if (dest / "segmenters.py").exists():
        return dest.resolve()
    cmd = [
        "git", "clone", "--depth", "1",
        "--branch", SOURCE_BRANCH, SOURCE_REPO, str(dest),
    ]
    if quiet:
        cmd.insert(2, "--quiet")
    subprocess.run(cmd, check=True)
    return dest.resolve()


# ── Dependencies ──────────────────────────────────────────────

def ensure_packages(quiet: bool = True) -> list[str]:
    """Install only what is actually missing.

    Colab ships most of these, so this is usually a no-op; a bare virtualenv
    gets the lot. Either way nothing is upgraded, because silently changing a
    working environment's package versions is the kind of help nobody asked for.
    """
    installed: list[str] = []
    for import_name, pip_name in REQUIRED.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            cmd = [sys.executable, "-m", "pip", "install", pip_name]
            if quiet:
                cmd.append("--quiet")
            subprocess.run(cmd, check=True)
            installed.append(pip_name)
    return installed


def stub_gradio() -> bool:
    """Register a minimal `gradio` module if the real one is absent.

    Returns True when a stub was installed. The toolkit only ever uses
    `gr.Error`, so a RuntimeError subclass is a faithful stand-in: the
    exceptions still raise, still carry their message, and still stop the cell.
    """
    try:
        importlib.import_module("gradio")
        return False
    except ImportError:
        pass

    class Error(RuntimeError):
        """Stand-in for gradio.Error — same behaviour outside a web UI."""

    stub = types.ModuleType("gradio")
    stub.Error = Error
    stub.__version__ = "0.0.0+nbtools-stub"
    sys.modules["gradio"] = stub
    return True


# ── The one call a notebook makes ─────────────────────────────

def setup(verbose: bool = True, source: str | pathlib.Path | None = None) -> Toolkit:
    """Make the toolkit importable and return its modules.

    Pass `source` to use a specific checkout — useful when testing a branch
    other than the pinned one.
    """
    installed = ensure_packages()

    if source is not None:
        source_path, cloned = pathlib.Path(source).resolve(), False
    else:
        found = find_source()
        if found is not None:
            source_path, cloned = found, False
        else:
            source_path, cloned = clone_source(), True

    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    stubbed = stub_gradio()

    modules = {name: importlib.import_module(name) for name in TOOLKIT_MODULES}

    toolkit = Toolkit(
        source_path=source_path,
        cloned=cloned,
        gradio_stubbed=stubbed,
        installed=installed,
        modules=modules,
    )
    if verbose:
        print(toolkit.summary())
    return toolkit
