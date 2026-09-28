#!/usr/bin/env python
"""
Psychrometric conversions, for deriving relative humidity from the fields the
reference datasets actually publish.

NEITHER REFERENCE PUBLISHES RH. ERA5 gives 2 m temperature and 2 m dewpoint;
CONUS404 gives 2 m temperature, a water-vapour mixing ratio and surface
pressure. Both have to be converted, and the conversion is where the errors
live, so it is isolated here and tested rather than inlined.

SATURATION IS TAKEN OVER LIQUID WATER AT ALL TEMPERATURES. That is the WMO
convention for surface observations and what station networks report, so a
model-observation comparison has to use the same one. Over ice below 0 C the
saturation pressure is lower, so an ice-based formula would give a HIGHER
relative humidity for the same air -- by up to about 10% at -20 C. Mixing the
two conventions would put a spurious cold-season bias into every comparison.

Bolton (1980) is used for saturation vapour pressure: better than 0.1% over
-35 to +35 C, which covers the domain. Magnus-Tetens and Buck differ from it by
under 0.3% in that range, so the choice is not a source of uncertainty here.
"""

from __future__ import annotations

import numpy as np

# Ratio of molar masses, water vapour to dry air.
EPSILON = 0.62198

# Physical bounds. Supersaturation is real in cloud but is not what a 2 m
# diagnostic reports, and values above ~105% in a surface product are almost
# always a conversion or rounding artefact.
RH_MIN, RH_MAX = 0.0, 100.0


def _to_celsius(t):
    """Kelvin or Celsius, decided by magnitude rather than by an argument.

    A caller passing the wrong unit is the most likely error here and it is
    silent: 288 C is physically absurd but arithmetically fine, and would give a
    relative humidity near zero everywhere. Surface air temperature is never
    near 100 in either unit, so the test is unambiguous.
    """
    t = np.asarray(t, dtype="float64") if not hasattr(t, "dims") else t
    return t - 273.15 if float(np.nanmax(t)) > 100 else t


def saturation_vapor_pressure(t):
    """Saturation vapour pressure over liquid water, in hPa. Bolton (1980)."""
    tc = _to_celsius(t)
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def rh_from_dewpoint(t, td, clip: bool = True):
    """Relative humidity in %, from temperature and dewpoint.

    Both may be in K or C; they are converted independently, so a caller who
    mixes units gets a wrong answer rather than a crash. Pass matching units.

    This is the ERA5 path: t2m and d2m.
    """
    rh = 100.0 * saturation_vapor_pressure(td) / saturation_vapor_pressure(t)
    return _clip(rh) if clip else rh


def rh_from_mixing_ratio(t, w, p, clip: bool = True):
    """Relative humidity in %, from temperature, mixing ratio and pressure.

    w is mass of vapour per mass of DRY air (kg/kg); p is in Pa or hPa, decided
    by magnitude. This is the CONUS404 path: T2, Q2, PSFC.

    WRF's Q2 is a mixing ratio, not a specific humidity, despite the name. The
    difference is w/(1+w) -- under 2% for surface air -- so confusing them
    produces a small, plausible-looking bias rather than an obvious failure.
    Use rh_from_specific_humidity if the field really is q.
    """
    p = np.asarray(p, dtype="float64") if not hasattr(p, "dims") else p
    p_hpa = p / 100.0 if float(np.nanmax(p)) > 2000 else p
    e = w * p_hpa / (EPSILON + w)                 # vapour pressure, hPa
    rh = 100.0 * e / saturation_vapor_pressure(t)
    return _clip(rh) if clip else rh


def rh_from_specific_humidity(t, q, p, clip: bool = True):
    """Relative humidity in %, from specific humidity (mass vapour / mass moist).

    LOCA2 publishes `huss` when it does not publish `hurs`, so this is the
    fallback path for the products themselves.
    """
    w = q / (1.0 - q)
    return rh_from_mixing_ratio(t, w, p, clip=clip)


def dewpoint_from_rh(t, rh):
    """Dewpoint in the same units as t. The inverse, for sanity checks."""
    tc = _to_celsius(t)
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    e = np.clip(rh, 1e-6, None) / 100.0 * es
    lg = np.log(e / 6.112)
    return 243.5 * lg / (17.67 - lg)


def _clip(rh):
    """Bound to [0, 100].

    Clipping rather than masking: values a little over 100 come from rounding in
    the source fields and from the formula's own error near saturation, and
    discarding those cells would preferentially remove the wettest conditions --
    exactly the ones a humidity analysis cares about. Anything far outside the
    range is a unit error and should be caught before it reaches here.
    """
    if hasattr(rh, "clip"):
        return rh.clip(RH_MIN, RH_MAX)
    return np.clip(rh, RH_MIN, RH_MAX)


def check(verbose: bool = True) -> bool:
    """Verify against hand-computable reference values."""
    cases = [
        # (T degC, Td degC, expected RH %) -- Bolton over LIQUID WATER.
        #
        # The two sub-freezing values are the ones to be careful with. Over ICE
        # the same pairs give about 42.5% and 38.5%, because saturation over ice
        # is lower, so an ice-referenced formula reports a HIGHER humidity for
        # identical air. Those are the numbers a reader half-remembering the
        # cold-season case will expect, and using them here would have quietly
        # asserted the opposite convention to the one this module implements
        # and that the observations follow.
        (20.0, 20.0, 100.0),
        (20.0, 10.0, 52.5),
        (30.0, 10.0, 28.9),
        (0.0, -10.0, 46.9),
        (-10.0, -20.0, 43.8),
    ]
    ok = True
    for t, td, want in cases:
        got = float(rh_from_dewpoint(t, td))
        good = abs(got - want) < 0.6
        ok &= good
        if verbose:
            print(f"  T={t:>6.1f}C Td={td:>6.1f}C -> RH {got:5.1f}% "
                  f"(expect {want:5.1f}) {'ok' if good else 'FAIL'}")

    # Kelvin and Celsius must agree.
    a = float(rh_from_dewpoint(293.15, 283.15))
    b = float(rh_from_dewpoint(20.0, 10.0))
    ok &= abs(a - b) < 1e-9
    if verbose:
        print(f"  K and C agree: {abs(a - b) < 1e-9}")

    # Mixing-ratio path must agree with the dewpoint path.
    t, td, p = 20.0, 10.0, 101325.0
    e = float(saturation_vapor_pressure(td))
    w = EPSILON * e / (p / 100.0 - e)
    c = float(rh_from_mixing_ratio(t, w, p))
    ok &= abs(c - b) < 0.2
    if verbose:
        print(f"  mixing-ratio path {c:.2f}% vs dewpoint {b:.2f}%: "
              f"{'ok' if abs(c - b) < 0.2 else 'FAIL'}")

    # Round trip through dewpoint.
    rt = float(dewpoint_from_rh(20.0, 52.5))
    ok &= abs(rt - 10.0) < 0.15
    if verbose:
        print(f"  round trip RH->Td: {rt:.2f}C (expect 10.0) "
              f"{'ok' if abs(rt - 10.0) < 0.15 else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    import sys
    print("psychrometric checks:")
    sys.exit(0 if check() else 1)
