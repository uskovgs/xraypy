import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
from astropy.coordinates import SkyCoord
import astropy.units as u


def plot_image(
    filename,
    title=None,
    caption=None,
    cmap="gray_r",
    stretch=None,
    interval=None,
    skycoords=None,
    radiuses=None,
    colors=None,
    figsize=(6.5, 6.5),
    origin="lower",
    fig=None,
    ax=None,
):
    """Plot a FITS image with WCS and optional sky markers.

    Parameters
    ----------
    filename : str or Path
        FITS file path (primary HDU used).
    title : str, optional
    cmap : str or Colormap, optional
    stretch : astropy.visualization.BaseStretch or None
    interval : astropy.visualization.Interval or None
    skycoords : SkyCoord or sequence, optional
        Sky coordinates to mark on the image.
    radiuses : number, Quantity, or sequence, optional
        Marker radii (pixels or astropy Quantity).
    colors : str or sequence, optional
    figsize : tuple, optional
    origin : {'lower','upper'} or int, optional

    Returns
    -------
    fig, ax
        Matplotlib Figure and Axes (WCS projection).

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord

    >>> from astropy import units as u

    >>> sc = SkyCoord(ra=10*u.deg, dec=0*u.deg)

    >>> fig, ax = plot_image('my.fits', title='My target', skycoords=sc,
    ...                     radiuses=2*u.arcsec)

    """
    if stretch is None:
        stretch = AsinhStretch(a=0.05)
    if interval is None:
        interval = PercentileInterval(99.5)

    with fits.open(filename) as hdul:
        hdu = hdul[0]
        img_data = np.squeeze(hdu.data)
        wcs = WCS(hdu.header)

    vmin, vmax = interval.get_limits(img_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=False)

    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection=wcs)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.imshow(img_data, cmap=cmap, norm=norm, origin=origin)

    if skycoords is not None:
        # skycoords -> list
        if isinstance(skycoords, SkyCoord):
            skycoords = [skycoords]
        else:
            skycoords = list(skycoords)

        radiuses = _broadcast_radiuses(radiuses, len(skycoords))
        colors = _broadcast_colors(colors, len(skycoords))

        for sc, r, c in zip(skycoords, radiuses, colors):
            _add_region(ax, wcs, sc, r, color=c)

    return fig, ax


def _broadcast_radiuses(radiuses, n):

    if radiuses is None:
        # дефолт: 5 пикселей
        return [5.0] * n

    # scalar number (pixels)
    if isinstance(radiuses, (int, float, np.number)):
        return [float(radiuses)] * n

    # scalar Quantity (sky units)
    if isinstance(radiuses, u.Quantity):
        return [radiuses] * n

    # sequence
    radiuses = list(radiuses)
    if len(radiuses) != n:
        raise ValueError(f"len(skycoords)={n} but len(radiuses)={len(radiuses)}")

    out = []
    for r in radiuses:
        if isinstance(r, u.Quantity):
            out.append(r)
        else:
            out.append(float(r))  # pixels
    return out


def _broadcast_colors(colors, n, default="red"):
    if colors is None:
        return [default] * n
    if isinstance(colors, str):
        return [colors] * n

    colors = list(colors)
    if len(colors) != n:
        raise ValueError(f"len(skycoords)={n} but len(colors)={len(colors)}")
    return colors


def _add_region(ax, wcs, skycoord, radius, color="red"):
    try:
        from regions import CircleSkyRegion, CirclePixelRegion, PixCoord
    except ImportError:
        raise ImportError("Install 'regions': pip install regions")

    if isinstance(radius, u.Quantity):
        # radius in sky units (e.g. arcsec)
        reg = CircleSkyRegion(skycoord, radius=radius)
        reg.to_pixel(wcs).plot(ax=ax, color=color, linewidth=0.8)
    else:
        # radius in pixels
        x, y = wcs.world_to_pixel(skycoord)
        reg = CirclePixelRegion(PixCoord(x, y), radius=float(radius))
        reg.plot(ax=ax, color=color, linewidth=0.8)
