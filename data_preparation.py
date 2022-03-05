#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from obspy import UTCDateTime
from glob import glob
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy.interpolate import interp1d


def prep_data(df, channel1, channel2, config, f_min, f_max, twin_min, twin_max, sta, cluster=3):
    # collect all the channels in an array
    data_array = []
    sigma_array = []
    t0_data = -1

    try:
        f_to_read = os.path.join(config["input_dir"],
                                 "{}.{}-{}.{}_stretching_list.csv".format(sta.upper(), channel1, sta.upper(), channel2))
        dvv_df_or = pd.read_csv(f_to_read)
        dvv_df_or.dropna(inplace=True)
    except:
        print("could not read {}".format(f_to_read))
        return [], [], [], [], [], False

    # #### Read and stitch the seismic data (dv/v)
    # selection by quality, frequency range, time window, etc
    dvv_df = dvv_df_or[dvv_df_or.t0_s == twin_min]
    dvv_df = dvv_df[dvv_df.t1_s == twin_max]
    dvv_df = dvv_df[dvv_df.f0_Hz == f_min]
    dvv_df = dvv_df[dvv_df.cluster == cluster]

    dat1 = stitch(dvv_df, config["reftimes"], plot=False)
    dat1 = dat1[dat1.s_cc_after >= config["min_quality"]]
    dat1.s_dvv -= dat1.s_dvv.mean()
    if len(dat1) < 10:
        print("Insufficient number of data points with quality > {}".format(config["min_quality"]))
        return [], [], [], [], [], False

    if t0_data == -1:
        t0_data = dat1.s_timestamps.min()
        t1_data = dat1.s_timestamps.max()

    # independent variables & observations
    dfsub = df[df.timestamps < min(config["t1"].timestamp, t1_data)].copy()
    dfsub = dfsub[dfsub.timestamps >= max(config["t0"].timestamp, t0_data)] 

    # interpolate onto df timestamps of met data
    f = interp1d(dat1.s_timestamps, dat1.s_dvv, "linear", bounds_error=False, fill_value=0)
    dfsub["dvv"] = f(dfsub.timestamps.values)
    g = interp1d(dat1.s_timestamps, dat1.s_err, "linear", bounds_error=False, fill_value=0)
    dfsub["error"] = g(dfsub.timestamps.values)
    h = interp1d(dat1.s_timestamps, dat1.s_cc_after, "linear", bounds_error=False, fill_value=0)
    dfsub["cc_after"] = h(dfsub.timestamps.values)
    # print("max dvv ", df["dvv"].max())
    # print("max error after Weaver ", df["error"].max())
    # print("min abs error after Weaver ", np.abs(df["error"]).min())
    dfsub["dvv"] /= 100.0  # convert from %
    dfsub["error"] /= 100.0  # convert from %
    rain_m = dfsub["rain"].values / 1000.0
    rain_m -= rain_m.mean()
    pressure_Pa = dfsub["pressure"].values
    temp_C = dfsub["Temp_C"].values
    dvv_obs = dfsub.dvv.values
    dvv_obs -= dvv_obs.mean()
    data_array.append(dvv_obs)
    sigma_array.append(dfsub.error.values)
    if len(data_array) == 1:
        data_array = np.array(data_array, ndmin=2)
        sigma_array = np.array(sigma_array, ndmin=2)
        sigma_array[sigma_array == 0.0] = 1.e-6

        t = dfsub.timestamps.values
        return t, data_array, sigma_array, rain_m, temp_C, pressure_Pa, True
    else:
        return [], [], [], [], [], False


def sta_to_metsta(sta):
    metstations = {"unm": "CCHS",
                   "UNM": "CCHS",
                   "cdmx_icvm": "ENP2",
                   "cdmx_aovm": "ENP6",
                   "cdmx_apvm": "CCHA",
                   "cdmx_bjvm": "ENP6",
                   "cdmx_cjvm": "CCHS",
                   "cdmx_covm": "ENP6",
                   "cdmx_ctvm": "ENP7",
                   "cdmx_gmvm": "ENP7",
                   "cdmx_mcvm": "CCHS",
                   "cdmx_mhvm": "ENP4",
                   "cdmx_mpvm": "ENP3",
                   "cdmx_thvm": "ENP1",
                   "cdmx_tlvm": "ENP3",
                   "cdmx_vrvm": "ENP7",
                   "cdmx_xcvm": "ENP1",
                   "MULU": "ENP7",
                   "MIXC": "ENP6",
                   "CIRE": "ENP6",
                   "ESTA": "ENP9",
                   "TEPE": "ENP1"}
    return(metstations[sta])


def kernels_map(sta):
    # map for kernels: some stations have the same type of site
    # (e.g bedrock stations are not
    # distinguished. For these, only one
    # set of kernels was computed.
    # This dictionary is used to find the right file for each site.
    kernels_map = {"unm": "unm",
                   "UNM": "unm",
                   "icvm": "icvm",
                   "aovm": "ipvm",
                   "apvm": "apvm",
                   "bjvm": "bjvm",
                   "cjvm": "ipvm",
                   "covm": "covm",
                   "ctvm": "ctvm",
                   "gmvm": "ipvm",
                   "mcvm": "ipvm",
                   "mhvm": "unm",
                   "mpvm": "ipvm",
                   "mzvm": "ipvm",
                   "ptvm": "ipvm",
                   "thvm": "thvm",
                   "tlvm": "ipvm",
                   "vrvm": "vrvm",
                   "xcvm": "xcvm",
                   "MULU": "MULU",
                   "MIXC": "MIXC",
                   "CIRE": "CIRE",
                   "ESTA": "ipvm",
                   "TEPE": "thvm"}

    return(kernels_map[sta])

def tempcosine(t, a, p):
    period = 365.25 * 86400. 
    dt = abs(mode(np.diff(t))[0][0])  # delta t, sampling (e.g. 1 day)
    t_ax = np.arange(len(t)) * dt
    return(a * np.cos(2. * np.pi * 1./period * t_ax + p))

def get_met_data(sta, metdatadir, time_resolution, do_plots=False):
    # meteorologic observations:
    # load data for a year or multiple years
    met_files = "{}/{}/*/*prep.csv".format(metdatadir, sta)
    files = glob(met_files)
    files.sort()

    df_w = pd.read_csv(files[0])
    for f in files[1:]:
        df2 = pd.read_csv(f)
        df_w = pd.concat((df_w, df2), ignore_index=True)

    # get timestamps from time strings
    datetimes = []
    for i in range(len(df_w)):
        try:
            datetime = UTCDateTime(df_w.Fecha_hora.values[i][0: 10] +
                                   "T" + df_w.Fecha_hora.values[i][-8:])
            datetimes.append(datetime.timestamp)
        except (ValueError, TypeError):
            datetimes.append(np.nan)
    df_w["timestamps"] = datetimes
    # df_w = df_w.dropna(subset=["timestamps", "Precipitacion_mm"]).reset_index()
    df_w["rain"] = df_w["Precipitacion_mm"]  #.apply(lambda x: re.sub(".0", "0", str(x))).astype(float)
    df_w["Temp_C"] = df_w["Temp_C"]  #df_w  .apply(lambda x: re.sub(".0", "0", str(x))).astype(float)
    df_w["pressure"] = df_w["Presion_bar_hPa"]  #.apply(lambda x: re.sub(".0", "0", str(x))).astype(float)
    df_w["rain"] = np.nan_to_num(df_w.rain.values)

    meantemp = np.nanmean(df_w.Temp_C.values)
    df_w["Temp_C"] = np.nan_to_num(df_w.Temp_C.values, nan=meantemp)
    meanpres = np.nanmean(df_w.pressure)
    df_w["pressure"] = np.nan_to_num(df_w.pressure.values, nan=meanpres)
    print(df_w.pressure.max(), df_w.pressure.min())

    # print("rain mean", df_w.rain.mean())
    # print("Temp mean", df_w.Temp_C.mean())
    # set up a new dataframe with a lower time resolution
    df = pd.DataFrame(columns=df_w.keys())
    df.timestamps = np.arange(np.nanmin(df_w.timestamps.values), np.nanmax(df_w.timestamps.values), int(time_resolution))
    # print("Old and new nr. timestamps: {}, {}".format(len(df_w), len(df)))
    print("MEAN RAIN: ", df_w.rain.mean())
    rain_avg = np.zeros(len(df))
    temp_avg = np.zeros(len(df))
    pres_avg = np.zeros(len(df))
    for ixt, timestamp in enumerate(df.timestamps.values):
        # rain_avg[ixt] = np.nan_to_num(df_w[(df_w.timestamps >= timestamp - 0.5 * time_resolution) &
        #                               (df_w.timestamps < timestamp + 0.5 * time_resolution)].rain.mean())
        # temp_avg[ixt] = np.nan_to_num(df_w[(df_w.timestamps >= timestamp - 0.5 * time_resolution) &
        #                               (df_w.timestamps < timestamp + 0.5 * time_resolution)].Temp_C.mean())
        rain_avg[ixt] = np.nan_to_num(df_w[(df_w.timestamps >= timestamp - time_resolution) &
                                      (df_w.timestamps < timestamp + time_resolution)].rain.mean())
        temp_avg[ixt] = np.nan_to_num(df_w[(df_w.timestamps >= timestamp - time_resolution) &
                                      (df_w.timestamps < timestamp + time_resolution)].Temp_C.mean())
        pres_avg[ixt] = np.nan_to_num(df_w[(df_w.timestamps >= timestamp - time_resolution) &
                                      (df_w.timestamps < timestamp + time_resolution)].pressure.mean())
    df["rain"] = rain_avg
    df["Temp_C"] = temp_avg
    df["pressure"] = pres_avg * 100.0  # convert from hectopascal to Pascal
    print(df.pressure.mean())
    print("MEAN RAIN adjusted: ", df.rain.mean())

    #plt.plot(df.timestamps, df.rain, "slateblue")
    #plt.plot(df.timestamps, df.Temp_C, "firebrick")
    #plt.show()

    if do_plots:
        plt.figure(figsize=(12, 3))
        plt.plot(df.timestamps, df.rain)
        tickstrs = []
        yrs = []
        newticks = []
        for ttstamp in df.timestamps:
            if UTCDateTime(ttstamp).strftime("%Y") not in yrs:
                tickstrs.append(UTCDateTime(ttstamp).strftime("%Y/%m"))
                newticks.append(ttstamp)
                yrs.append(UTCDateTime(ttstamp).strftime("%Y"))
        a, _ = plt.xticks(newticks, tickstrs, rotation=90)
        plt.ylabel("Rain (mm)")
        plt.title("Precipitacion at station {}".format(sta))
        plt.tight_layout()
        plt.savefig("rain_{}.png".format(sta))
        plt.close()

        plt.figure(figsize=(12, 3))
        plt.plot(df.timestamps, df.pressure)
        tickstrs = []
        yrs = []
        newticks = []
        for ttstamp in df.timestamps:
            if UTCDateTime(ttstamp).strftime("%Y") not in yrs:
                tickstrs.append(UTCDateTime(ttstamp).strftime("%Y/%m"))
                newticks.append(ttstamp)
                yrs.append(UTCDateTime(ttstamp).strftime("%Y"))
        a, _ = plt.xticks(newticks, tickstrs, rotation=90)
        plt.ylabel("Pressure (Pa)")
        plt.title("Atm. pressure at station {}".format(sta))
        plt.tight_layout()
        plt.savefig("atmpres_{}.png".format(sta))
        plt.close()

    return(df)


def adjust_dvv_offset(d1, d2, n=10, plot=False):
    # The following function adjusts the vertical offset in dv/v between time
    # segments with different references so that the mean
    # square error between the measurement segments is minimized
    overlapping_timestamps = []
    i = -1
    while len(overlapping_timestamps) < n:
        overlapping_timestamps.extend(list(np.intersect1d(d1.timestamps[i:], d2.timestamps)))
        overlapping_timestamps = list(set(overlapping_timestamps))
        i -= 1
        if abs(i) > len(d1):
            break

    if len(overlapping_timestamps) > 0:
        aixs = [np.where(d1.timestamps.values == otst)[0][0] for otst in
                overlapping_timestamps]
        a = d1.dvv.values[aixs]  # - offset_add
        bixs = [np.where(d2.timestamps.values == otst)[0][0] for otst in
                overlapping_timestamps]
        b = d2.dvv.values[bixs]
        offset = (b - a).mean()  # for comparison
        possible_offsets = np.linspace((b - a).min(), (b - a).max(), 100)
        l2_norm = []
        for i, testoffset in enumerate(possible_offsets):
            l2_norm.append(np.sum(0.5 * (a - (b - testoffset)) ** 2))
        optimal_offset = possible_offsets[np.argmin(np.array(l2_norm))]

        if plot:
            plt.figure()
            h1 = plt.scatter(overlapping_timestamps,
                             a, edgecolor="k",)
            h2 = plt.scatter(overlapping_timestamps,
                             b,edgecolor="k")

            plt.title("Stitching illustrated")
            for ixta, tstref in enumerate(overlapping_timestamps):
                plt.vlines(x=tstref, ymin=a[ixta], ymax=b[ixta],
                           linestyles="--",
                           color="rebeccapurple",)
            plt.grid()
            plt.legend([h1, h2], ["Reference 1", "Reference 2"])
            plt.ylabel("dv/v %")
            xticklabels = []
            locs, labels = plt.xticks()
            for l in locs:
                xticklabels.append(UTCDateTime(l).strftime("%Y/%m/%d"))
            plt.xticks(locs, xticklabels, rotation=45)
            plt.tight_layout()
    else:
        # print("No overlap")
        offset = 0.
        optimal_offset = 0.0
    return(optimal_offset)


def stitch(dat, times_refs, plot=False):
    # this function does the book-keeping in merging segments of the dv/v curve
    # that have been obtained with different references.
    # using the adjust_dvv_offset function, it determines the optimal offset
    # that minimizes the difference between equivalent samples of dv/v on curves with different
    # reference period, and then gives back a pandas dataframe with the merged dv/v curve and
    # related data.
    dat1 = pd.DataFrame(columns=["s_dvv", "s_cc_before", "s_cc_after", "s_timestamps"])
    dat1["s_dvv"] = np.zeros(len(dat))
    dat1["s_cc_before"] = np.zeros(len(dat))
    dat1["s_cc_after"] = np.zeros(len(dat))
    dat1["s_timestamps"] = np.zeros(len(dat))
    dat1["s_err"] = np.zeros(len(dat))
    ix = 0
    off = 0.
    if len(times_refs) > 1:
        for ixt, t in enumerate(times_refs):
            if ixt > 0:
                dat_until = dat[dat.timestamps < times_refs[ixt]]
                off += adjust_dvv_offset(dat_until[dat_until.tag == ixt - 1], dat[dat.tag == ixt], n=10, plot=plot)
                plot = False
            else:
                off = 0.0
            if ixt < len(times_refs) - 1 and ixt > 0:
                cond = (dat.tag == ixt) & (dat.timestamps >= t) & (dat.timestamps < times_refs[ixt + 1])
            elif ixt == 0:
                cond = (dat.tag == ixt) & (dat.timestamps < times_refs[ixt + 1])
            else:
                cond = (dat.tag == ixt) & (dat.timestamps >= t)

            tp = dat[cond].copy()
            dat1["s_dvv"].values[ix: ix + len(tp)] = tp.dvv.values - off
            dat1["s_cc_before"].values[ix: ix + len(tp)] = tp.cc_before.values
            dat1["s_cc_after"].values[ix: ix + len(tp)] = tp.cc_after.values
            dat1["s_timestamps"].values[ix: ix + len(tp)] = tp.timestamps.values
            dat1["s_err"].values[ix: ix + len(tp)] = tp.dvv_err.values
            ix += len(tp)
    else:
        dat1["s_dvv"] = dat.dvv.values
        dat1["s_cc_before"] = dat.cc_before.values
        dat1["s_cc_after"] = dat.cc_after.values
        dat1["s_timestamps"] = dat.timestamps.values
        dat1["s_err"] = dat.dvv_err.values
    dat1.s_dvv -= dat1.s_dvv.mean()
    return(dat1)