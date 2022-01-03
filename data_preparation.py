#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from obspy import UTCDateTime
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import re


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
                   "mvcm": "ipvm",
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
    df_w = df_w.dropna(subset=["timestamps", "Precipitacion_mm"]).reset_index()
    df_w["rain"] = df_w["Precipitacion_mm"].apply(lambda x: re.sub(".0", "0", str(x))).astype(float)
    df_w["Temp_C"] = df_w["Temp_C"].apply(lambda x: re.sub(".0", "0", str(x))).astype(float)

    # set up a new dataframe with a lower time resolution
    df = pd.DataFrame(columns=df_w.keys())
    df.timestamps = np.arange(np.nanmin(df_w.timestamps.values), np.nanmax(df_w.timestamps.values), int(time_resolution))
    print("Old and new nr. timestamps: {}, {}".format(len(df_w), len(df)))
    rain_avg = np.zeros(len(df))
    temp_avg = np.zeros(len(df))
    for ixt, timestamp in enumerate(df.timestamps.values):
        rain_avg[ixt] = np.nan_to_num(df_w[(df_w.timestamps >= timestamp - 0.5 * time_resolution) &
                                      (df_w.timestamps < timestamp + 0.5 * time_resolution)].rain.mean())
        temp_avg[ixt] = df_w[(df_w.timestamps >= timestamp - 0.5 * time_resolution) &
                             (df_w.timestamps < timestamp + 0.5 * time_resolution)].Temp_C.mean()
    df["rain"] = rain_avg
    df["Temp_C"] = temp_avg
    df = df.dropna(subset=["rain"]).reset_index()

    if do_plots:
        plt.figure(figsize=(12, 3))
        plt.plot(df.timestamps, df.Precipitacion_mm)
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
        print("No overlap")
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
    dat1.s_dvv -= dat1.s_dvv.mean()
    return(dat1)