import numpy as np
from math import sqrt, pi
from model_tools import func_rain, func_quake, func_healing, func_temp1, func_lin, func_pseudo_SSW

def evaluate_model1(ind_vars, params):
    t = ind_vars[0]
    z = ind_vars[1]
    kernel_vs = ind_vars[2]
    rho = ind_vars[3]
    dp_rain = ind_vars[4]
    temperature = ind_vars[5]
    nsm_samples = ind_vars[6]

    p0 = 10. ** params[0]
    tau_max = 10. ** params[1]
    drop = params[2]
    slope = params[3] / (365. * 86400)
    const = params[4]
    shift = params[5] * 30.0 * 86400.0
    scale = params[6]

    dv_rain = func_rain([z, dp_rain, rho, kernel_vs], [p0])
    dv_temp = func_temp1([t, temperature, nsm_samples], [shift, scale])
    dv_quake = func_healing([t], [tau_max, drop])
    dv_lin = func_lin([t], [slope, const])
    #print(dv_rain.max(), dv_temp.max(), dv_quake.max(), dv_lin.max())
    #print(dv_rain.mean(), dv_temp.mean(), dv_quake.mean(), dv_lin.mean())
    return(dv_rain + dv_temp + dv_quake + dv_lin)

def evaluate_model4(ind_vars, params):
    t = ind_vars[0]
    rain_m = ind_vars[1]
    temperature = ind_vars[2]
    nsm_samples = ind_vars[3]

    phi = 10. ** params[0]
    a = params[1]
    tau_max = 10. ** params[2]
    drop = params[3]
    slope = params[4] / (365. * 86400.)
    const = params[5]
    shift = params[6] * (30. * 86400.)
    scale = params[7]

    dv_rain = func_pseudo_SSW([t, rain_m], [phi, a])
    dv_temp = func_temp1([t, temperature, nsm_samples], [shift, scale])
    dv_quake = func_healing([t], [tau_max, drop])
    dv_lin = func_lin([t], [slope, const])
    # print(dv_rain.max(), dv_temp.max(), dv_quake.max(), dv_lin.max())
    return(dv_rain + dv_temp + dv_quake + dv_lin)


def evaluate_model2(ind_vars, params):
    t = ind_vars[0]
    z = ind_vars[1]
    kernel_vs = ind_vars[2]
    rho = ind_vars[3]
    dp_rain = ind_vars[4]
    temperature = ind_vars[5]
    nsm_samples = ind_vars[6]

    p0 = 10. ** params[0]
    drop = params[1]
    recovery = params[2]
    slope = params[3] / (365. * 86400)
    const = params[4]
    shift = 30. * 86400. * params[5] - ind_vars[7] / 2.0
    scale = params[6]

    dv_rain = func_rain([z, dp_rain, rho, kernel_vs], [p0])
    dv_quake = func_quake([t], [drop, recovery])
    dv_lin = func_lin([t], [slope, const])
    dv_temp = func_temp1([t, temperature, nsm_samples], [shift, scale])

    return(dv_rain + dv_temp + dv_quake + dv_lin)


def evaluate_model3(ind_vars, params):
    t = ind_vars[0]
    z = ind_vars[1]
    kernel_vs = ind_vars[2]
    rho = ind_vars[3]
    dp_rain = ind_vars[4]
    temperature = ind_vars[5]
    nsm_samples = ind_vars[6]

    p0 = 10. ** params[0]
    slope = params[1] / (365. * 86400)
    const = params[2]
    shift = 30. * 86400. * params[3] - ind_vars[5] / 2.0
    scale = params[4]

    dv_rain = func_rain([z, dp_rain, rho, kernel_vs], [p0])
    dv_lin = func_lin([t], [slope, const])
    dv_temp = func_temp1([t, temperature, nsm_samples], [shift, scale])

    return(dv_rain + dv_temp + dv_lin)

def set_bounds(config):
    lower_bounds = []
    upper_bounds = []
    for param in config["list_params"]:
        print(param)
        lower_bounds.append(config["bounds_" + param][0])
        upper_bounds.append(config["bounds_" + param][1])
    bounds = [lower_bounds, upper_bounds]
    return(bounds)


def get_mcmc_bounds(config):
    # same as above, but using scaling or log for MCMC
    lower_bounds = []
    upper_bounds = []
    for i, p in enumerate(config["list_params"]):
        if p == "waterlevel_p" or p == "phi" or p == "tau_max":
            # bounds p0
            lower_bounds.append(np.log10(config["bounds_{}".format(p)][0]))
            upper_bounds.append(np.log10(config["bounds_{}".format(p)][1]))
        elif p == "shift":
            # bounds shift
            lower_bounds.append((config["bounds_{}".format(p)][0]) / (30. * 86400.))
            upper_bounds.append((config["bounds_{}".format(p)][1]) / (30. * 86400.))
        elif p == "slope":
            lower_bounds.append((config["bounds_{}".format(p)][0]) * (365. * 86400.))
            upper_bounds.append((config["bounds_{}".format(p)][1]) * (365. * 86400.))
        else:
            lower_bounds.append(config["bounds_{}".format(p)][0])
            upper_bounds.append(config["bounds_{}".format(p)][1])
    if config["use_logf"]:
        lower_bounds.append(config["bounds_logf"][0])
        upper_bounds.append(config["bounds_logf"][1])
    elif config["use_g"]:
        lower_bounds.append(config["bounds_g"][0])
        upper_bounds.append(config["bounds_g"][1])
        
    #print("BOUNDS:")
    #print([lower_bounds, upper_bounds])
    bounds = [lower_bounds, upper_bounds]

    return(bounds)


def get_initial_position_from_mlmodel(params_mod, config):
    init_pos = params_mod.copy()

    for i, p in enumerate(config["list_params"]):
        if p == "waterlevel_p" or p == "phi" or p == "tau_max":
            init_pos[i] = np.log10(init_pos[i])
        elif  p == "slope":
            init_pos[i] *= (365. * 86400)
        elif p == "shift":
            init_pos[i] /= (30. * 86400.)
    if config["use_logf"]:
        init_pos = np.concatenate((init_pos, np.array([-2]))) # append log f
    if config["use_g"]:
        init_pos = np.concatenate((init_pos, np.array([1.01]))) # append g
    print("Initial position is around ", init_pos)
    return(init_pos)

# uniform PRIOR for emcee
def uniform_prior_for_emcee(params, bounds):
    for i, p in enumerate(params):
        if p < bounds[0][i]:
            return(-np.inf)
        if p > bounds[1][i]:
            return(-np.inf)
    return(0.0)

# neg. log. LIKELIHOOD for emcee
def log_likelihood_for_emcee(params, ind_vars, data, data_err, fmodel, error_is_underestimated_logf=False, error_is_underestimated_g=False):
    ll = 0.0

# get the synthetics for these parameters
    if error_is_underestimated_g:
        mparams = params[:-1]
        g = params[-1]
    elif error_is_underestimated_logf:
        mparams = params[:-1]
        log_f = params[-1]        
    else:
        mparams = params

    if fmodel == 'model1':
        synth = evaluate_model1(ind_vars, mparams)
    elif fmodel == "model2":
        synth = evaluate_model2(ind_vars, mparams)
    elif fmodel == "model4":
        synth = evaluate_model4(ind_vars, mparams)

    # data could be several dimensions
    if np.ndim(data) == 1:
        data = np.array(data, ndmin=2)
        data_err = np.array(data_err, ndmin=2)

    for ixd, d in enumerate(data):
        if error_is_underestimated_g:
            sigma2 = (g * data_err[ixd]) ** 2
        elif error_is_underestimated_logf:
            sigma2 = (data_err[ixd] + np.exp(log_f)) ** 2 #+ np.exp(2 * log_f)
        else:
            sigma2 = data_err[ixd] ** 2

        res = synth - d
        # log_f parameter is used to account for possible underestimate of observational uncertainty.
        # uncertainty (data_err) is otherwise based on Weaver et al
        ll += -0.5 * np.sum(res ** 2 / sigma2) - np.sum(np.log(np.sqrt(2. * pi * sigma2)))

    return ll

# neg. log. probability for emcee
def log_probability_for_emcee(params, ind_vars, bounds, data, cov, fmodel, error_is_underestimated_logf, error_is_underestimated_g):

    prior = uniform_prior_for_emcee(params, bounds)
    llh = log_likelihood_for_emcee(params, ind_vars, data, cov, fmodel, error_is_underestimated_logf, error_is_underestimated_g)
    if not np.isfinite(prior):
        return(-np.inf)
    else:
        return(llh + prior)
