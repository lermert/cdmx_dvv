import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import pandas as pd
from scipy.special import erf, erfc
from scipy.stats import mode
from glob import glob
from ruido.models.models import model_cdmx_discrete
from scipy.signal import convolve


def model_SW_dG(p_in, c1, c2):
    # this function returns the derivative of the Wilson smooth model
    # for determining the sensitivity of stress with respect to pore pressure
    p_in[0] = p_in[1]
    return(c1 * c2 / 3. * np.power(c2 * p_in, -2./3.))


def roeloffs(t, rain, r, B_skemp, nu, diff,
             rho=1000.0, g=9.81,
             waterlevel=1.e-6,
             model="both"):
    # this function returns the convolution of the impulse response of the solution to the 1-D 
    # homogeneous coupled poroelastic problem, derived by E. Roeloffs (1988), with a time series
    # of rain that here is used as "hydraulic head change"
    alpha = B_skemp * (1 + nu) / (3. - 3. * nu)
    dp = rho * g * rain

    r = np.tile(r, len(t)).reshape((len(t), len(r)))
    t -= t.min()
    t[0] = waterlevel

    diffterm = np.divide(r.T,  np.sqrt(4 * diff * t))

    if model == "drained":
        irf = (1 - alpha) * erfc(diffterm)
    elif model == "undrained":
        irf = alpha
    elif model == "both":
        irf = alpha + (1 - alpha) * erfc(diffterm)
    else:
        raise ValueError("Model must be \"drained\" or \"undrained\" or \"both\".")
    P = np.zeros((len(t), r.shape[1]))
    for i in range(r.shape[1]):
        P[:, i] = convolve(irf[i, :], dp, mode="same")
    return P

def func_lin(independent_vars, params):
    # linear trend
    t = independent_vars[0]
    slope = params[0]
    const = params[1]

    y = slope * t + const
    return(y)

def func_quake(independent_vars, params):
    # Heaviside step function and logarithmic recovery
    # used to describe earthquake dv/v following Nakata & Snieder, 2011
    t = independent_vars[0]
    a = params[0]
    b = params[1]
    s0 = 1.
    tquake = UTCDateTime("2017,09,19").timestamp
    heavi = np.heaviside(t-tquake, s0)
    log = np.zeros(len(t))
    ix = np.where(t > tquake)[0]
    log[ix] = a * np.log((t[ix] - tquake))
    return heavi * log - heavi * b

def get_rainload_p(t, z, rain_m, station, drained_undrained_both="both",
                   use_clearyrice=True, diff_in=None, B_skemp=1.0,
                   precomp_rain=None):
    # This function determines the response to precipitation in terms of pore pressure,
    # using Roeloff's impulse response to the 1-D coupled poroel. problem as defined above.
    z0 = 10
    # geology:
    vs, vp, rho, qs, qp, nu, B, nu_u = model_cdmx_discrete(z0, station, output="poroelastic")

    # use inferred values for B and nu_u?
    if use_clearyrice:
        B_skemp = B
        nu = nu_u
    p =roeloffs(t, rain_m, z, B_skemp, nu, diff_in, model=drained_undrained_both)
    return(p)

def func_rain(independent_vars, params):
    # This function does the bookkeeping for predicting dv/v from pore pressure change.
    z = independent_vars[0]
    dz = z[1] - z[0]
    kernel = independent_vars[1]
    p = independent_vars[2]
    rho_avg = independent_vars[3]
    mu_avg = independent_vars[4]
    dp_rain = independent_vars[5]

    # params: sensitivity, diffusivity, station, method, kernel
    c1 = params[0]
    c2 = params[1]

    dmudP = model_SW_dG(p, c1, c2)
    dbetadP = 1. / np.sqrt(rho_avg) * 1. / (2. * np.sqrt(mu_avg)) * dmudP
    stress_sensitivity = dbetadP
    # pore pressure is compressional
    dv_rain = np.dot(-dp_rain, stress_sensitivity * kernel * dz)

    return(dv_rain)
