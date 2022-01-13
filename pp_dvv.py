#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import seaborn as sns
from obspy import UTCDateTime
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import optimize
from model_tools import func_rain, func_lin, func_quake, parse_input, func_sciopt, get_rainload_p
from data_preparation import stitch, get_met_data, kernels_map
import os
import sys

# Velocity changes model in two or three steps:
# 1. Predict pore pressure change plus load from precipitation
# 2. Model change in shear wave velocity due to rain
# 3. Translate shear wave velocity into Rayleigh and / or Love wave velocity
# 4. Combine rain model with logarithmic recovery of velocity
# for earthquake and linear trend for subsidence / groundwater loss

# depth sensitivity
def get_c_from_str(fname):
    return(float(fname.split("=")[-1]))

config = parse_input(sys.argv[1])
t0_0 = config["t0"]
t1_0 = config["t1"]
T0 = 5

# preparations
if not os.path.exists(config["output_dir"]):
    os.makedirs(config["output_dir"])
output_filename = "model_{}_{}.csv".format(config["tag"], config["inversion_type"])
output_table = pd.DataFrame(columns=["sta", "ch1", "ch2", "f0_Hz", "t0_s", "cluster", "wave", "mode",
                                     "diff", "slope", "offset", "recov_eq", "drop_eq",
                                     "sig_slope", "sig_offset", "sig_recov_eq", "sig_drop_eq",
                                      "CC"])
# load meteo data
df = get_met_data(config["metstation"], config["meteo_data_dir"],
                  config["time_resolution"], do_plots=config["do_plots"])
df = df[df.timestamps < config["t1"]]
df = df[df.timestamps >= config["t0"]]

# loop over stations
for ixsta, sta in enumerate(config["stas"]):
    print(sta)
    # loop over clusters
    for cluster in config["clusters"]:
        station = "cdmx_" + kernels_map(config["stas"][ixsta])

        for ixf, f_min in enumerate(config["f_mins"]):
            f_max = 2. * f_min
            twins_min = [-1. / f_min * 8., -1. / f_min * 8., 1. / f_min * 4., 1. / f_min * 8.]
            twins_max = [-20. / f_min, -10. / f_min, 10. / f_min, 20. / f_min]

            for ixtw, twin_min in enumerate(twins_min):
                twin_max = twins_max[ixtw]
                # collect all the channels in an array
                data_array = []
                sigma_array = []
                n_channels = 0
                t0_data = -1
                for ixc, cp in enumerate(config["channelpairs"]):
                    channel1 = cp[0]
                    channel2 = cp[1]
                    print("trying to read ", os.path.join(config["input_dir"],
                        "{}.{}-{}.{}_stretching_list.csv".format(sta.upper(),
                             channel1, sta.upper(), channel2)))
                    try:
                        dvv_df_or = pd.read_csv(os.path.join(config["input_dir"],
                        "{}.{}-{}.{}_stretching_list.csv".format(sta.upper(),
                        channel1, sta.upper(), channel2)))
                       
                        # print(dvv_df_or)
                        dvv_df_or.dropna(inplace=True)
                        # plt.close()
                    except:
                        print("could not read.")
                        continue
                
                    # #### Read and stitch the seismic data (dv/v)
                    # selection by quality, frequency range, time window, etc
                    dvv_df = dvv_df_or[dvv_df_or.t0_s == twin_min].copy()
                    dvv_df = dvv_df[dvv_df.t1_s == twin_max]
                    dvv_df = dvv_df[dvv_df.f0_Hz == f_min]
                    dvv_df = dvv_df[dvv_df.cluster == cluster]
                    

                    dat1 = stitch(dvv_df, config["reftimes"], plot=config["do_plots"])
                    # print(dvv_df.timestamps)
                    plt.close()
                    dat1 = dat1[dat1.s_cc_after >= config["min_quality"]]
                    dat1.s_dvv -= dat1.s_dvv.mean()
                    if len(dat1) < 10:
                        continue
                    if t0_data == -1:
                        t0_data = dat1.s_timestamps.min()
                        t1_data = dat1.s_timestamps.max()

                    # interpolate onto df timestamps of met data
                    f = interp1d(dat1.s_timestamps, dat1.s_dvv, "linear", bounds_error=False, fill_value=0)
                    df["dvv"] = f(df.timestamps.values)
                    g = interp1d(dat1.s_timestamps, dat1.s_err, "linear", bounds_error=False, fill_value=0)
                    df["error"] = g(df.timestamps.values)
                    # print("max dvv ", df["dvv"].max())
                    # print("max error after Weaver ", df["error"].max())
                    # print("min abs error after Weaver ", np.abs(df["error"]).min())
                    df["dvv"] /= 100.0  # convert from %
                    df["error"] /= 100.0  # convert from %

                    # independent variables & observations
                    dfsub = df[df.timestamps < min(config["t1"].timestamp, t1_data)].copy()  # df[df.timestamps < min(t1, dat1.s_timestamps.max())].copy()
                    dfsub = dfsub[dfsub.timestamps >= max(config["t0"].timestamp, t0_data)]  # dfsub[dfsub.timestamps >= max(t0, dat1.s_timestamps.min())]
                    #print("nr. selected times: ", len(dfsub))
                    # print("*"*80)
                    rain_m = dfsub["rain"].values / 1000.0
                    rain_m -= rain_m.mean()
                    dvv_obs = dfsub.dvv.values
                    dvv_obs -= dvv_obs.mean()
                    data_array.append(dvv_obs)
                    sigma_array.append(dfsub.error.values)
                if len(data_array) == 0: continue
                data_array = np.array(data_array, ndmin=2)
                sigma_array = np.array(sigma_array, ndmin=2)
                sigma_array[sigma_array == 0.0] = 1.e-6

                # print("number of nans in precipitation data: ", np.isnan(rain_m).sum())
                try:
                    t = dfsub.timestamps.values
                except NameError:
                    continue


                kernels = glob("{}/kernels_{}.{}.f={}*".\
                    format(config["kernel_dir"], config["wavepol"], kernels_map(sta), f_min))
                kernels.sort(key=get_c_from_str)
                try:
                    k = np.loadtxt(kernels[config["mode_nr"]], skiprows=3)
                except:
                    print("problem finding kernel")
                    continue
                z_k = k[:, 0]
                vs_k = k[:, 10]
                vs_k += k[:, 11]
                # interpolate kernel to the depth vector we are using
                fint = interp1d(np.abs(z_k - 6371000.0), vs_k, bounds_error=False, fill_value=0, kind="nearest")  # interpolate to the z defined here
                K_vs = fint(config["z"])

                for diff in config["tdiffs"]:
                    p = get_rainload_p(t, config["z"], rain_m,
                        station, diff_in=diff, drained_undrained_both=config["roeloffs_method"])
                    

                    # choose the optimizer and run optimization
                    if config["inversion_type"] == "nonlinear_lsq":
                        bounds = ([config["bounds_slope"][0],
                                   config["bounds_const"][0],
                                   config["bounds_recov_eq"][0],
                                   config["bounds_drop_eq"][0],
                                   #config["bounds_m"][0],
                                   #config["bounds_k"][0],
                                   #config["bounds_yb"][0]
                                   ],
                                  [config["bounds_slope"][1],
                                   config["bounds_const"][1],
                                   config["bounds_recov_eq"][1],
                                   config["bounds_drop_eq"][1],])
                                   #config["bounds_m"][1],
                                   #config["bounds_k"][1],
                                   #config["bounds_yb"][1],] )


                        model_to_fit = lambda t, slope, const, recov_eq, drop_eq:\
                        func_sciopt(t, slope, const, recov_eq, drop_eq, # m, k, yb,
                                    p, K_vs, config["z"], station, T0,
                                    n_channels=len(config["channelpairs"])
                                    )

                        # Fitting
                        if config["use_bounds"]:
                            params_mod, covariance_mod =\
                                optimize.curve_fit(model_to_fit, t,
                                                   data_array.ravel(), sigma=sigma_array.ravel(), bounds=bounds)
                        else:
                            params_mod, covariance_mod =\
                                optimize.curve_fit(model_to_fit, t,
                                                   data_array.ravel(), sigma=sigma_array.ravel())

                        cc_dat_mod = np.corrcoef(data_array.ravel(), model_to_fit(t, *params_mod))[0][1]
                        print(params_mod)

                        sigmas = np.sqrt(np.diag(covariance_mod))
                        newcol = [sta, channel1, channel2, f_min, twin_min, cluster,
                                  config["wavepol"], config["mode_nr"], diff, *params_mod, *sigmas, cc_dat_mod]

                    output_table.loc[len(output_table)] = newcol
                    if diff == min(config["tdiffs"]):
                        plt.figure(figsize=(6, 3))
                        plt.subplot(111)
                        for i in range(len(config["channelpairs"])):
                            h1, = plt.plot(t, data_array[i, :], "0.7", linewidth=1.5, alpha=0.7)
                        h2, = plt.plot(t, model_to_fit(t, *params_mod)[0: len(t)], "darkorange", linewidth=1.5, zorder=40)

                        #hq, = plt.plot(t, func_quake(t, *params_mod[3:6]), ":", zorder=30, color="rebeccapurple", alpha=0.7)
                        #h3, = plt.plot(t, t*params_mod[1] + params_mod[2], "--", zorder=20, color="rebeccapurple", alpha=0.7)
                        #params_rain = params_mod[0: 2]
                        #h4, = plt.plot(t, params_mod[0] * np.dot((p), K_vs * dz),"-.", zorder=10, color="rebeccapurple", alpha=0.7)
                        ticks = plt.xticks()
                        tickstrs = []
                        newticks = []
                        for tick in ticks[0]:
                            tickstrs.append(UTCDateTime(float(tick)).strftime("%Y/%m"))
                            newticks.append(float(tick))
                        a, _ = plt.xticks(newticks, tickstrs, rotation=90)
                        # plt.grid()
                        # plt.xlim(config["t0"].timestamp, config["t1"].timestamp)
                        # plt.ylim([-0.02, 0.02])
                        plt.ylabel("dv/v (-)")
                        plt.legend([h1, h2], ["observation", "model", "linear trend", "earthquake", "precipitation"],
                                    loc="lower center", ncol=3, fontsize="small")
                        plt.title("Station: %s, Corr.coeff: %4.3f" %(sta, round(cc_dat_mod, 3)))

                        # plt.subplot(122)
                        # plt.plot(t, dvv_obs, "k", alpha=0.7, linewidth=1)
                        # plt.plot(t, (dvv_obs - model_to_fit(t, *params_mod)))
                        # ticks = plt.xticks()
                        # tickstrs = []
                        # newticks = []
                        # for tick in ticks[0]:
                        #     tickstrs.append(UTCDateTime(float(tick)).strftime("%Y/%m"))
                        #     newticks.append(float(tick))
                        # a, _ = plt.xticks(newticks, tickstrs, rotation=90)
                        # plt.grid()
                        # plt.legend(["observation", "residual"])
                        # plt.ylabel("residual dv/v (-)")
                        # plt.title("data-model residuals")

                        plt.tight_layout()
                        plt.savefig("{}/dvvmodel_{}.{}-{}.cl{}.{}Hz.{}s.png".format(config["output_dir"], sta, channel1,
                            channel2, cluster,
                            f_min, twin_min), dpi=300)
                        plt.show()
                        plt.close()

                        print("CC between data and model: ", cc_dat_mod)
    output_table.to_csv(os.path.join(config["output_dir"], output_filename))
