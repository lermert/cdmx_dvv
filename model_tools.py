import numpy as np
from obspy import UTCDateTime
from scipy.special import erf, erfc
from scipy.stats import mode
from velocity_models import model_cdmx_discrete
from scipy.fftpack import next_fast_len
import yaml
from model_tools_kurama import logheal_llc, GWL_SSW06
from scipy.interpolate import interp1d
from glob import glob
from data_preparation import kernels_map

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

    if config["use_logf"] and config["use_g"]:
        raise ValueError("Either use_logf or use_g must be false, both cannot be used at the same time! Please edit input file.")
    return config

#################################################
# Hydrology: 1-D poroelastic response to rainfall
#################################################

def model_SW_dsdp(p_in, waterlevel=100.):
    # 1 / vs  del v_s / del p: Derivative of shear wave velocity to effective pressure
    # identical for Walton smooth model and Hertz-Mindlin model
    # input: 
    # p_in (array of int or float): effective pressure (hydrostatic - pore)
    # waterlevel: to avoid 0 division at the free surface. Note that results are sensitive to this parameter.
    # output: 1/vs del vs / del p
    #try:
    p_in += waterlevel
    sens = 1. / (6. * p_in)
    # except ValueError:
    #     p_in = np.tile(p_in, len(waterlevel)).reshape(len(waterlevel), len(p_in))
    #     p_in += np.tile(waterlevel, p_in.shape[1]).reshape(p_in.shape[1], len(waterlevel)).T
    #     sens = 1. / (6. * p_in)

    return(sens)

def roeloffs_1depth(t, rain, r, B_skemp, nu, diff,
                    rho, g, waterlevel, model, nfft=None):
    # evaluate Roeloff's response function for a specific depth r
    # input:
    # t: time vector in seconds
    # rain: precipitation time series in m
    # r: depth in m
    # B_skemp: Skempton's coefficient (no unit)
    # nu: Poisson ratio (no unit)
    # diff: Hydraulic diffusivity, m^2/s
    # rho: Density of water (kg / m^3)
    # g: gravitational acceleration (N / kg)
    # waterlevel: to avoid zero division at the surface. Results are not sensitive to the choice of waterlevel
    # model: drained, undrained or both (see Roeloffs, 1988 paper)
    # output: Pore pressure time series at depth r

    # use nfft to try an increase convolution speed
    if nfft is None:
        nfft = len(t)

    dp = rho * g * rain
    dt = abs(mode(np.diff(t))[0][0])  # delta t, sampling (e.g. 1 day)
    diffterm = 4. * diff * np.arange(len(t)) * dt
    diffterm[0] = waterlevel
    diffterm = r / np.sqrt(diffterm)
    
    resp = erf(diffterm)
    rp = np.zeros(nfft)
    rp[0: len(resp)] = resp
    P_ud = np.convolve(rp, dp, "full")[0: len(dp)]
    
    resp = erfc(diffterm)
    rp = np.zeros(nfft)
    rp[0: len(resp)] = resp
    P_d = np.convolve(rp, dp, "full")[0: len(dp)]
    if model == "both":
        P = P_d + B_skemp * (1 + nu) / (3. - 3. * nu) * P_ud
    elif model == "drained":
        P = P_d
    elif model == "undrained":
        P = B_skemp * (1 + nu) / (3. - 3. * nu) * P_ud
    else:
        raise ValueError("Unknown model for Roeloff's poroelastic response. Model must be \"drained\" or \"undrained\" or \"both\".")
    return P

def roeloffs(t, rain, r, B_skemp, nu, diff, rho=1000.0, g=9.81, waterlevel=1.e-12, model="both"):
    s_rain = np.zeros((len(t), len(r)))
    fftN = next_fast_len(len(t))
    for i, depth in enumerate(r):
        p = roeloffs_1depth(t, rain, depth, B_skemp, nu, diff,
                            rho, g, waterlevel, model, nfft=fftN)
        s_rain[:, i] = p
    return(s_rain)

def get_rainload_p(t, z, rain_m, station, diff_in=1.e-2, drained_undrained_both="both",
                   ):
    # This function determines the response to precipitation in terms of pore pressure,
    # using Roeloff's impulse response to the 1-D coupled poroel. problem as defined above.
    z0 = 10
    # geology:
    vs, vp, rho, qs, qp, nu, B, nu_u = model_cdmx_discrete(z0, station, output="poroelastic")
    p = roeloffs(t, rain_m, z, B, nu_u, diff_in, model=drained_undrained_both)
    return(p)


def func_rain(independent_vars, params):
    # This function does the bookkeeping for predicting dv/v from pore pressure change.
    z = independent_vars[0]
    dp_rain = independent_vars[1]
    rhos = independent_vars[2]
    kernel = independent_vars[3]
    dz = z[1] - z[0]

    waterlevel = params[0]

    p = np.zeros(len(z))
    p[1: ] = np.cumsum(rhos * 9.81 * dz)[:-1]  # overburden / hydrostatic pressure
    stress_sensitivity = model_SW_dsdp(p, waterlevel)
    dv_rain = np.dot(-dp_rain, stress_sensitivity * kernel * dz)

    return(dv_rain)

def func_rain1(independent_vars, params):
    # This function does the bookkeeping for predicting dv/v from pore pressure change.
    z = independent_vars[0]
    dp_rain = independent_vars[1]
    rhos = independent_vars[2]
    kernel = independent_vars[3]
    pressure = independent_vars[4]
    dz = z[1] - z[0]

    fac = params[0]

    p = np.zeros(len(z))
    p[1: ] = np.cumsum(rhos * 9.81 * dz)[:-1]  # overburden / hydrostatic pressure

    dv_rain = np.zeros(len(dp_rain))
    for i in range(len(dp_rain)):
        stress_sensitivity = model_SW_dsdp(p, pressure[i])
        dv_rain[i] = np.dot(-dp_rain[i], (stress_sensitivity * kernel * dz))

    return(fac * dv_rain)

#################################################
# Hydrology: scaled inferred ground water level
#################################################

def func_pseudo_SSW(independent_vars, params):
    t = independent_vars[0]
    rain_m = independent_vars[1]

    phi = params[0]  # this effectively takes the role of a scaling factor.
    a = params[1]

    dv_rain = -1. * GWL_SSW06(t, rain_m, phi, a)
    return(dv_rain)

#################################################
# Linear velocity change
#################################################

def func_lin(independent_vars, params):
    # linear trend
    t = independent_vars[0]
    slope = params[0]
    const = params[1]

    dv_y = slope * (t - t.min()) + const
    return(dv_y)

#################################################
# Co- and postseismic velocity change
#################################################

# a: logarithmic, bounding recovery to original value of velocity
# drop_eq is the drop
# a controls how fast it recovers
def func_quake(independent_vars, params, time_quake="2017,09,19,18,14,00"):
    # Heaviside step function and logarithmic recovery
    # used to describe earthquake dv/v following Nakata & Snieder, 2011
    # also compatible with Snieder et al. (2017) for tau_min << t << tau_max
    t = independent_vars[0]
    drop_eq = params[0]
    a = params[1]

    tquake = UTCDateTime(time_quake).timestamp
    heavi = np.heaviside(t-tquake, 1.)
    
    log = np.zeros(len(t))
    ix = np.where(t > tquake)[0]
    log[ix] = a * np.log((t[ix] - tquake))
    dv_quake = log - heavi * drop_eq
    return dv_quake

# b: healing
def func_healing(independent_vars, params, time_quake="2017,09,19,18,14,00"):
    # full implementation of Snieder's healing model from Snieder et al. 2017
    # but faster
    # (c) by Kurama Okubo
    t = independent_vars[0]

    t_low = np.linspace(t.min(), t.max(), 100)
    tau_min = 0.1
    tau_max = params[0]
    drop_eq = params[1]
    tquake = UTCDateTime(time_quake).timestamp
    

    tax = t_low - tquake
    ixt = tax > 0
    tax[~ixt] = 0.0

    dv_quake_low = np.zeros(len(tax))

    # separate function accelerated by c and low level callback for scipy quad
    dv_quake_low[ixt] = [logheal_llc(tt, tau_min, tau_max, drop_eq) for tt in tax[ixt]]
    dv_quake_low /= np.log(tau_max/tau_min)
    # reinterpolate
    f = interp1d(t_low, dv_quake_low, bounds_error=False)
    dv_quake = f(t)
    return(dv_quake)



#################################################
# Thermoelastic effect following Richter et al., 2015
#################################################

def temperature_dv_richter(z, nu, alpha, kappa, toe, k, T_surface, w):
    gamma = np.sqrt(w / (2. * kappa))
    exp1 = 2. * np.exp(-(1 + 1.j) * gamma * z)
    exp2 = -(1. + nu) * (1. - nu) * k / gamma * np.exp(-k * z)
    fac = alpha * (1. + nu) / (1. - nu) * toe * T_surface  # T_surface is a cosine describing surface temperature
    return(fac * np.abs(exp1 * exp2))

def func_temp(independent_vars, params):
    raise NotImplementedError("Probably not correct")
    t = independent_vars[0]
    z = independent_vars[1]

    nu = independent_vars[2]
    T_surface = independent_vars[3]
    kernel = independent_vars[4]
    dz = z[1] - z[0]

    k = params[0]
    toe = params[1]

    alpha_th = 1.e-5  # berger 1975
    kappa = 1.e-6
    w = 2. * np.pi / (364.25 * 86400.)

    dvv_T = np.zeros((len(t), len(z)))

    for i, zz in enumerate(z):
        dvv_T[:, i] = temperature_dv_richter(zz, nu[i], alpha_th,
                                             kappa, toe, k, T_surface, w)
    dv_thermoel = np.dot(dvv_T, kernel * dz)
    return(dv_thermoel)



def func_temp1(independent_vars, params):

    t = independent_vars[0]
    T_surface = independent_vars[1]
    nsm_samples = independent_vars[2]

    tsurf = np.zeros(len(T_surface) + 2 * nsm_samples)
    tsurf[nsm_samples: -nsm_samples] = T_surface
    tsurf[0: nsm_samples] = tsurf[nsm_samples]
    tsurf[-nsm_samples:] = tsurf[-nsm_samples-1]
    T_surface = np.convolve(tsurf, np.ones(nsm_samples)/nsm_samples, mode='same')[nsm_samples: -nsm_samples]

    shift = params[0]
    scale = params[1]

    f = interp1d(t, T_surface, kind="nearest", bounds_error=False, fill_value="extrapolate")

    t_new = t - shift
    temp_new = scale * f(t_new)
    # print(temp_new.max())

    return(temp_new)

#################################################
# Bookkeeping to get material parameters
#################################################
def get_rho_nu(z, sta):
    rhos = []
    nus = []
    for zz in z:
        vs, vp, rho = model_cdmx_discrete(zz, sta)[0:3]
        rhos.append(rho)
        ab = vp / vs
        nus.append((ab ** 2 - 2) / (2 * ab**2 - 2))
    rhos = np.array(rhos)
    nus = np.array(nus)
    return(rhos, nus)

#################################################
# Bookkeeping to set up the full model
#################################################

def func_sciopt(t, list_models, list_vars, list_params, n_channels):

    model = np.zeros(len(t))
    for ixm, m in enumerate(list_models):
        model += m(list_vars[ixm], list_params[ixm])


    return(np.tile(model, n_channels))

# depth sensitivity
def get_c_from_str(fname):
    return(float(fname.split("=")[-1]))


def get_sens_kernel(config, sta, f_min):
    kernels = glob("{}/kernels_{}.{}.f={}*".\
        format(config["kernel_dir"], config["wavepol"], kernels_map(sta), f_min))
    kernels.sort(key=get_c_from_str)
    try:
        k = np.loadtxt(kernels[config["mode_nr"]], skiprows=3)
        z_k = k[:, 0]
        vs_k = k[:, 10]
        vs_k += k[:, 11]
        # interpolate kernel to the depth vector we are using
        fint = interp1d(np.abs(z_k - 6371000.0), vs_k,
                        bounds_error=False, fill_value=0, kind="nearest")  # interpolate to the z defined here
        K_vs = fint(config["z"])
        if config["suppress_shallow_sensitivity"]:
            K_vs[config["z"] < config["depth_to_sens"]] = 0.0
        success = True

    except:
        print("problem finding kernel fo {}, {} Hz".format(sta, f_min))
        success = False

    return(K_vs, success)