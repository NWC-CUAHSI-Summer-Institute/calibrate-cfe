"""
Modules to compute PET using different methods

Author: Abhinav Gupta (Created: 4 Feb 2026)

"""

import pandas as pd
import math
import numpy as np

def net_radiation(srad, tmin, tmax, elev, lat, J, ea, albedo):
    """
    Computing net radiation (pandas-based)

    Parameters
    ----------
    srad : pd.Series
        Shortwave solar radiation at earth surface (W/m2)
    tmin, tmax : pd.Series
        Minimum and maximum temperature (°C)
    elev : float
        Elevation above mean sea level (m)
    lat : float
        Latitude (decimal degrees)
    J : pd.Series or int
        Julian day of year
    ea : pd.Series
        Actual vapor pressure (Pa)
    albedo : float

    Returns
    -------
    Rn_net : pd.Series
        Net radiation (W/m2)
    """

    # Emissivity
    e = 1.0

    # Stefan–Boltzmann constant (W m-2 K-4)
    sigma = 5.67*10**(-8)

    # Solar constant (MJ m-2 min-1)
    Gsc = 0.0820

    # Convert vapor pressure from Pa to kPa
    ea = ea / 1000.0

    # Latitude in radians (scalar)
    phi = lat*math.pi / 180.0

    # Inverse relative Earth–Sun distance
    dr = 1.0 + 0.033 * np.cos(2.0 * math.pi * J / 365.0)

    # Solar declination
    delta = 0.409 * np.sin(2.0 * math.pi * J / 365.0 - 1.39)

    # Sunset hour angle
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))

    # Extraterrestrial radiation (MJ m-2 day-1)
    Ra = (
        (24.0 * 60.0 / math.pi)
        * Gsc
        * dr
        * (
            omega_s * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
        )
    )

    # Convert Ra to W m-2
    Ra = Ra / 0.024 / 3.6

    # Clear-sky solar radiation
    Rs0 = (0.75 + 2*10**(-5) * elev) * Ra

    # Net shortwave radiation
    Rns = srad * (1.0 - albedo)

    # Relative radiation (capped at 1)
    relative_radiation = srad / Rs0
    relative_radiation[relative_radiation > 1.0] = 1.0

    # Net longwave radiation
    Rnl = (
        e
        * sigma
        * 0.5
        * ((tmin + 273.15) ** 4 + (tmax + 273.15) ** 4)
        * (0.34 - 0.14 * ea**0.5)
        * (1.35 * relative_radiation - 0.35)
    )

    # Net radiation
    Rn_net = Rns - Rnl

    return Rn_net


def priestley_taylor_fixed_alpha(R_n, G, T, alpha):
    """
    Priestley–Taylor potential evapotranspiration (fixed alpha)

    Parameters
    ----------
    R_n : np.array
        Net radiation (W/m2)
    G : np.array
        Ground heat flux (W/m2)
    T : np.array
        Air temperature (°C)
    alpha : float
        Priestley–Taylor coefficient

    Returns
    -------
    PET : np.array
        Potential evapotranspiration (mm/day)
    """

    # Water density (kg/m3)
    pw = 1000.0

    # Saturation vapor pressure (kPa)
    es = 0.611 * np.exp(17.27 * T / (237.3 + T))

    # Slope of saturation vapor pressure curve (kPa/°C)
    delta = 4098.0 * es / (T + 237.3) ** 2

    # Latent heat of vaporization (kJ/kg)
    lv = 2501.0 - 2.37 * T

    # Psychrometric constant (kPa/°C)
    C_p = 1.005     # kJ kg-1 °C-1
    p = 101.3       # kPa
    gamma = C_p * p / 0.622 / lv

    # Priestley–Taylor available energy (W/m2)
    PET_E = alpha * (R_n - G) * delta / (delta + gamma)

    # Convert energy to evapotranspiration
    # W/m2 → m/s → mm/day
    PET = PET_E / lv / pw / 1000.0
    PET = PET * 1000.0 * 86400.0

    # PET cannot be negative
    PET = PET
    PET[PET < 0.0] = 0.0

    return PET