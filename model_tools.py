import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import pandas as pd
from scipy.special import erf, erfc
from scipy.stats import mode
from glob import glob
from ruido.models.models import model_cdmx_discrete
from scipy.signal import fftconvolve
import yaml

def parse_input(configfile):

    config = yaml.safe_load(open(configfile))
    config["stas"] = [st1.split("_")[1] for st1 in config["stations"]]  # short name for kernel files
    z = np.linspace(config["z0"], config["z1"], config["nz"])  # depth to consider
    config["dz"] = z[1] - z[0]
    config["z"] = z
    config["t0"] = UTCDateTime(config["t0"])
    config["t1"] = UTCDateTime(config["t1"])
    refs = []
    for ref in config["references"]:
        refs.append(UTCDateTime(ref))
    config["reftimes"] = refs
    return config

def model_SW_dG(p_in, c1, c2):
    # this function returns the derivative of the Wilson smooth model
    # for determining the sensitivity of stress with respect to pore pressure
    p_in[0] = p_in[1] # avoid 0 division at the surfaace (no overburden pressure)
    return(c1 * c2 / 3. * np.power(c2 * p_in, -2./3.))

def model_SW_dsdp(p_in, vs):
    p_in[0] = p_in[1] # avoid 0 division at the surfaace (no overburden pressure)
    return(vs / (6. * p_in))

def roeloffs_1depth(t, rain, r, B_skemp, nu, diff, rho, g, waterlevel, model):
    dp = rho * g * rain
    dt = abs(mode(np.diff(t))[0][0])  # delta t, sampling (e.g. 1 day)
    diffterm = 4. * diff * np.arange(len(t)) * dt
    diffterm[0] = waterlevel
    diffterm = r / np.sqrt(diffterm)
    resp = erf(diffterm)
    rp = np.zeros(len(resp) * 2)
    rp[len(resp): ] = resp


    P_ud = fftconvolve(dp, rp, "same")
    resp = erfc(diffterm)
    rp = np.zeros(len(resp) * 2)
    rp[len(resp): ] = resp
    P_d = fftconvolve(dp, rp, "same")
    if model == "both":
        P = P_d + B_skemp * (1 + nu) / (3. - 3. * nu) * P_ud
    else:
        raise ValueError("Cookies were too plentiful in Bern.")
    return P

def roeloffs(t, rain, r, B_skemp, nu, diff, rho=1000.0, g=9.81, waterlevel=1.e-6, model="both"):
    s_rain = np.zeros((len(t), len(r)))
    for i, depth in enumerate(r):
        p = roeloffs_1depth(t, rain, depth, B_skemp, nu, diff, rho, g, waterlevel, model)
        s_rain[:, i] = p
    return(s_rain)


def func_lin(independent_vars, params):
    # linear trend
    t = independent_vars[0]
    slope = params[0]
    const = params[1]

    dv_y = slope * t + const
    return(dv_y)

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
    dv_quake = heavi * log - heavi * b
    return dv_quake

def get_rainload_p(t, z, rain_m, station, diff_in=1.e-2, drained_undrained_both="both",
                   ):
    # This function determines the response to precipitation in terms of pore pressure,
    # using Roeloff's impulse response to the 1-D coupled poroel. problem as defined above.
    z0 = 10
    # geology:
    vs, vp, rho, qs, qp, nu, B, nu_u = model_cdmx_discrete(z0, station, output="poroelastic")
    p = roeloffs(t, rain_m, z, B, nu_u, diff_in, model=drained_undrained_both)
    return(p)

def temperature_dv_tsai(t, z, m, mu, k, nu, alpha_th, T0, kappa, yb, w):
    A_t = (1. + nu) / (1. - nu) * k * alpha_th * T0 *\
     np.sqrt(kappa / w) * np.exp(-np.sqrt(2/(2.*kappa) * yb))\
     * np.cos(w * t - np.sqrt(w / (2. * kappa)) * yb - np.pi / 4.)

    return m / mu * A_t * np.exp(-k*z)

def func_temp(independent_vars, params):
    t = independent_vars[0]
    z = independent_vars[1]
    mu = independent_vars[2]
    nu = independent_vars[3]
    T0 = independent_vars[4]
    kernel = independent_vars[5]
    dz = z[1] - z[0]


    m = params[0]
    k = params[1]
    yb = params[2]

    alpha_th = 1.e-5  # berger 1975
    kappa = 1.e-6
    w = 2. * np.pi / (364.25 * 86400.)
    
    
    dvv_T = np.zeros((len(t), len(z)))
    for i, zz in enumerate(z):
        dvv_T[:, i] = temperature_dv_tsai(t, zz, m, mu[i], k, nu[i], alpha_th, T0, kappa, yb, w)
    dv_thermoel = np.dot(dvv_T, kernel * dz)
    # return(dv_thermoel)

    return(np.zeros(len(t)))

def func_rain(independent_vars):
    # This function does the bookkeeping for predicting dv/v from pore pressure change.
    z = independent_vars[0]
    dp_rain = independent_vars[1]
    rhos = independent_vars[2]
    mus = independent_vars[3]
    kernel = independent_vars[4]
    dz = z[1] - z[0]

    p = np.cumsum(rhos * 9.81 * np.diff(z, prepend=0))
    stress_sensitivity = model_SW_dsdp(p, np.sqrt(mus / rhos))
    dv_rain = np.dot(-dp_rain, stress_sensitivity * kernel * dz)

    return(dv_rain)


def func_sciopt(t, slope, const, recov_eq, drop_eq, # m, k, yb,
                dp_rain, K_vs, z, sta, T0, n_channels):
    
    rhos = []
    mus = []
    nus = []
    for zz in z:
        vs, vp, rho = model_cdmx_discrete(zz, sta)[0:3]
        rhos.append(rho)
        mus.append(vs ** 2 * rho)
        ab = vp / vs
        nus.append((ab ** 2 - 2) / (2 * ab**2 - 2))
    rhos = np.array(rhos)
    mus = np.array(mus)
    nus = np.array(nus)
    
    vars_rain = [z, dp_rain, rhos, mus, K_vs]

    #vars_temp = [t, z, mus, nus, T0, K_vs]
   # params_temp = [m, k, yb]

    vars_quake = [t]
    params_quake = [recov_eq, drop_eq]

    vars_lin = [t]
    params_lin = [slope, const]

    term1 = func_rain(vars_rain)#, params_rain)
   # term2 = func_temp(vars_temp, params_temp)
    term3 = func_quake(vars_quake, params_quake)
    term4 = func_lin(vars_lin, params_lin)

    return(np.tile((term1 + term3 + term4), n_channels))
