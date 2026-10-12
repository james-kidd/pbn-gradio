"""
The seam between pixel space and the world.

Every measurement this toolkit produces — a hectare, a corridor length, a
dollar — is derived from a pixel count multiplied by an assumed ground
resolution. That assumption was previously scattered inline through several
modules, each recomputing `ground_resolution * resize_scale` and each free to
get it wrong independently. One of them did.

Centralising it here does two things. It makes the conversion testable in one
place, and more importantly it creates the single attachment point for real
georeferenced data. A bare uploaded image gets a `GeoContext` built from a
user-stated scale, with no CRS and no world coordinates — honest about being
ungeoreferenced. A GeoTIFF or a WMS tile from a provincial imagery service
gets one built from its affine transform and CRS, and every downstream
calculation becomes real without a single change to the code that consumes it.

The distinction is tracked rather than hidden: `georeferenced` says which case
you are in, and exports carry it so nobody mistakes pixel coordinates for
survey coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeoContext:
    """Pixel ↔ world mapping for one working image.

    `transform` is an affine in the GDAL/rasterio ordering
    (a, b, c, d, e, f), mapping pixel (col, row) to world (x, y):

        x = a*col + b*row + c
        y = d*col + e*row + f

    For an ungeoreferenced image it is None and only `m_per_px` is meaningful.
    """
    m_per_px: float
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    source: str = "assumed scale"

    # ── Constructors ──────────────────────────────────────────

    @classmethod
    def from_scale(cls, m_per_px: float, resize_scale: float = 1.0) -> "GeoContext":
        """A bare image with a user-stated ground resolution.

        `resize_scale` is original_dimension / working_dimension. Resizing for
        speed changes what one pixel covers, and forgetting that multiplier
        silently rescales every downstream area and cost — which is exactly the
        bug this constructor exists to make impossible to repeat.
        """
        return cls(m_per_px=float(m_per_px) * float(resize_scale))

    @classmethod
    def from_affine(
        cls,
        transform: tuple[float, float, float, float, float, float],
        crs: str,
        resize_scale: float = 1.0,
        source: str = "georeferenced raster",
    ) -> "GeoContext":
        """A georeferenced raster — the real-geodata path.

        Ground resolution is taken from the transform's pixel width. This
        assumes a projected CRS in metres; a geographic CRS (degrees) would need
        reprojecting first, and is rejected rather than silently mis-scaled.
        """
        a, b, c, d, e, f = transform
        if crs and crs.upper().replace(" ", "") in ("EPSG:4326", "WGS84"):
            raise ValueError(
                "A geographic CRS (degrees) cannot be measured in metres. "
                "Reproject to a projected CRS — a UTM zone or a provincial "
                "standard such as EPSG:3347 — before building a GeoContext."
            )
        scaled = tuple(v * resize_scale for v in (a, b, d, e))
        return cls(
            m_per_px=float(abs(a) * resize_scale),
            crs=crs,
            transform=(scaled[0], scaled[1], c, scaled[2], scaled[3], f),
            source=source,
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def georeferenced(self) -> bool:
        return self.transform is not None and self.crs is not None

    # ── Conversions ───────────────────────────────────────────

    def area_ha(self, pixels: int | float) -> float:
        """Pixel count to hectares."""
        return float(pixels) * (self.m_per_px ** 2) / 10_000.0

    def length_m(self, pixels: float) -> float:
        """Pixel distance to metres."""
        return float(pixels) * self.m_per_px

    def pixel_to_world(self, row: float, col: float) -> tuple[float, float]:
        """Pixel centre to world coordinates.

        Returns (col, row) unchanged when ungeoreferenced, so callers get pixel
        coordinates rather than a plausible-looking lie. Check `georeferenced`
        before treating the result as a survey position.
        """
        if self.transform is None:
            return (float(col), float(row))
        a, b, c, d, e, f = self.transform
        return (a * col + b * row + c, d * col + e * row + f)

    # ── Reporting ─────────────────────────────────────────────

    def describe(self) -> str:
        if self.georeferenced:
            return (
                f"{self.source} · {self.crs} · {self.m_per_px:.4f} m/px · "
                "world coordinates available"
            )
        return (
            f"{self.source} · {self.m_per_px:.4f} m/px · "
            "not georeferenced (coordinates are pixels)"
        )

    def export_note(self) -> str:
        """The line that goes into an export so nobody mistakes it for survey data."""
        if self.georeferenced:
            return f"Coordinates are {self.crs}. One pixel is {self.m_per_px:.4f} m."
        return (
            "Coordinates are image pixels, NOT geographic. "
            f"One pixel is {self.m_per_px:.4f} m on the ground, by assumption."
        )
