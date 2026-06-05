"""Shared helpers: geohash decode + feature loading."""
import numpy as np
import pandas as pd

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def decode(gh):
    """Decode a geohash string to (lat, lon) cell center."""
    latr = [-90.0, 90.0]
    lonr = [-180.0, 180.0]
    even = True
    for c in gh:
        cd = _BASE32.index(c)
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lonr[0] + lonr[1]) / 2
                if cd & mask:
                    lonr[0] = mid
                else:
                    lonr[1] = mid
            else:
                mid = (latr[0] + latr[1]) / 2
                if cd & mask:
                    latr[0] = mid
                else:
                    latr[1] = mid
            even = not even
    return (latr[0] + latr[1]) / 2, (lonr[0] + lonr[1]) / 2


def geo_coords(geohashes):
    """Return a dict geohash -> (lat, lon)."""
    return {g: decode(g) for g in pd.unique(geohashes)}
