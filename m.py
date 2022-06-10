#!/usr/bin/env python
# coding: utf-8
import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy import optimize
from model_tools import parse_input, func_sciopt, get_sens_kernel, get_rho_phi, get_rainload_p, func_rain1, \
func_lin, func_quake, func_healing, func_rain, func_temp1, func_pseudo_SSW, get_temperature_z, func_temp, func_rain_rhophi
from inv import set_bounds, get_mcmc_bounds, get_initial_position_from_mlmodel, evaluate_model6, evaluate_model0,\
log_probability_for_emcee, evaluate_model2, evaluate_model1, evaluate_model4, evaluate_model3, evaluate_model5,\
evaluate_model0a, evaluate_modelf, evaluate_modelfq, evaluate_modelf_noquake, evaluate_model_rhophi
from data_preparation import get_met_data, kernels_map, prep_data, sta_to_metsta
import os
import sys
from multiprocessing import Pool
import corner
from obspy import UTCDateTime

# turn off underlying numpy paralellisation
os.environ["OMP_NUM_THREADS"] = "1"

####################################################################################################
#  script
####################################################################################################

config = parse_input(sys.argv[1])
t0_0 = config["t0"]
t1_0 = config["t1"]
qtimes = [UTCDateTime(tst).timestamp for tst in config["quakes"]]
print(qtimes)

# preparations
if not os.path.exists(config["output_dir"]):
    os.makedirs(config["output_dir"])
os.system("cp {} {}".format(sys.argv[1], config["output_dir"]))
output_filename = "model_{}.csv".format(config["inversion_type"])


# loop over stations
for ixsta, sta in enumerate(config["stas"]):

    metsta = sta_to_metsta(sta)
    # load meteo data
    df = get_met_data(metsta, config["meteo_data_dir"],
                      config["time_resolution"], do_plots=config["do_plots"])
    df = df[df.timestamps < config["t1"]]
    df = df[df.timestamps >= config["t0"]]

    
    # loop over clusters
    for cluster in config["clusters"]:

        # difference between "sta" and "station" parameter: station is like a longer name
        # that is used by the velocity model script
        station = "cdmx_" + kernels_map(config["stas"][ixsta])
        if config["use_gradient_velocity_model"]:
            station += "_gradient"

        for ixf, f_min in enumerate(config["f_mins"]):

            for cp in config["channelpairs"]:
                channel1 = cp[0]
                channel2 = cp[1]
                f_max = 2. * f_min

                if channel1 == channel2:
                    twins_min = [1. / f_min * 4., 1. / f_min * 8.]
                    twins_max = [10. / f_min, 20. / f_min]
                else:
                    twins_min = [-20. / f_min, -10. / f_min, 1. / f_min * 4., 1. / f_min * 8.]
                    twins_max = [-8. / f_min, -4. / f_min, 10. / f_min, 20. / f_min]

                for ixtw, twin_min in enumerate(twins_min):
                    twin_max = twins_max[ixtw]
                    
                    print(sta, channel1, channel2, cluster, f_min, twin_min)


                    t, data_array, sigma_array, rain_m, temp_C, pressure_Pa, success = \
                        prep_data(df, channel1, channel2, config, f_min, f_max, twin_min, twin_max, sta)
                    if not success:
                        continue

                    for dd in data_array:
                        dd -= dd[0]

                    for diff_in in config["tdiffs"]:

                        for diff_in_temp in config["tdiffs_thermal"]:

                            # output file basename
                            datafob = "{}_{}-{}_{}cl_{}Hz_{}s".format(sta, channel1, channel2, cluster, f_min, twin_min)
                            fob = "{}_{}-{}_{}cl_{}Hz_{}s_{}m2ps".format(sta, channel1, channel2, cluster, f_min, twin_min, diff_in)
                            if diff_in_temp is not None:
                                fob += "_{}mp2s".format(diff_in_temp)

                            # don't repeat calculations / start off again where was interrupted
                            if os.path.exists(config["output_dir"] + "/tau_{}.npy".format(fob)):
                                print("{} was already inverted, continuing...".format(fob))
                                continue

                            # tau, autocorrelation time: Diagnostic for convergence
                            have_tau = False

                            # find z array
                            # if not config["use_variable_depth"]:
                            #     modelz = config["z"]
                            # else:
                            #     if f_min == 0.5:
                            #         modelz = np.linspace(0., 2000., config["nz"])
                            #     elif f_min == 1.0:
                            #         modelz = np.linspace(0., 1000., config["nz"])
                            #     elif f_min == 2.0:
                            #         modelz = np.linspace(0., 500., config["nz"])
                            #     elif f_min == 4.0:
                            #         modelz = np.linspace(0., 250., config["nz"])
                            #     else:
                            #         raise ValueError("Not defined, set use_variable_depth to false in config file")
                            modelz = config["z"]
                            K_vs, success = get_sens_kernel(config, sta, (f_max - f_min) / 2. + f_min, modelz)
                            if not success:
                                continue

                            #tempz = np.linspace(0., 50., 501)
                            tempz = modelz
                            K_vs_temp = K_vs
                            #K_vs_temp, success = get_sens_kernel(config, sta, (f_max - f_min) / 2. + f_min, tempz)

                            # get material parameters
                            rhos, phis = get_rho_phi(modelz, station)



                            ###############################################################################
                            dp_rain = get_rainload_p(t, modelz, rain_m,
                                station, diff_in=diff_in, drained_undrained_both=config["roeloffs_method"])

                            if diff_in_temp is not None:
                                dp_temp = get_temperature_z(t, temp_C, tempz,
                                diff_in_temp, config["smoothing_temperature_n_samples"])


                            # get maximum likelihood as starting model
                            if config["model"] == "model_rhophi":
                                model_to_fit = lambda t, waterlevel_p, phi, tau_max, drop_eq, slope, const, tsens:\
                                               func_sciopt(t,
                                               list_models=[func_rain_rhophi, func_healing, func_lin, func_temp],
                                               list_vars=[[modelz, dp_rain, rhos, K_vs], [t], [t], [t, tempz, K_vs_temp, dp_temp]],
                                               list_params=[[waterlevel_p, phi], [tau_max, drop_eq], [slope, const], [tsens]],
                                               n_channels=1)
                            elif config["model"] == "modelf":
                                model_to_fit = lambda t, waterlevel_p, tau_max, drop_eq, slope, const, tsens:\
                                               func_sciopt(t,
                                               list_models=[func_rain, func_healing, func_lin, func_temp],
                                               list_vars=[[modelz, dp_rain, rhos, phis, K_vs], [t], [t], [t, tempz, K_vs_temp, dp_temp]],
                                               list_params=[[waterlevel_p], [tau_max, drop_eq], [slope, const], [tsens]],
                                               n_channels=1)
                            elif config["model"] == "modelf_noquake":
                                model_to_fit = lambda t, waterlevel_p, slope, const, tsens:\
                                               func_sciopt(t,
                                               list_models=[func_rain, func_lin, func_temp],
                                               list_vars=[[modelz, dp_rain, rhos, phis, K_vs], [t], [t, tempz, K_vs_temp, dp_temp]],
                                               list_params=[[waterlevel_p], [slope, const], [tsens]],
                                               n_channels=1)
                            # elif config["model"] == "modelfa":
                            #     model_to_fit = lambda t, waterlevel_p, tau_max, drop_eq, tsens:\
                            #                    func_sciopt(t,
                            #                    list_models=[func_rain, func_healing, func_temp],
                            #                    list_vars=[[modelz, dp_rain, rhos, phis, K_vs], [t], [t, tempz, K_vs_temp, dp_temp]],
                            #                    list_params=[[waterlevel_p], [tau_max, drop_eq], [tsens]],
                            #                    n_channels=1)
                            elif config["model"] == "modelfq":
                                model_to_fit = lambda t, waterlevel_p, drop_eq, recovery, slope, const, tsens:\
                                               func_sciopt(t,
                                               list_models=[func_rain, func_quake, func_lin, func_temp],
                                               list_vars=[[modelz, dp_rain, rhos, phis, K_vs], [t], [t], [t, tempz, K_vs_temp, dp_temp]],
                                               list_params=[[waterlevel_p], [drop_eq, recovery], [slope, const], [tsens]],
                                               n_channels=1)
                            elif config["model"] == "model1" or config["model"] == "model0":
                                model_to_fit = lambda t, waterlevel_p, tau_max, drop_eq, slope, const, shift, scale:\
                                               func_sciopt(t,
                                               list_models=[func_rain, func_healing, func_lin, func_temp1],
                                               list_vars=[[modelz, dp_rain, rhos, K_vs], [t], [t], [t, temp_C, config["smoothing_temperature_n_samples"]]],
                                               list_params=[[waterlevel_p], [tau_max, drop_eq], [slope, const], [shift, scale]],
                                               n_channels=1)
                            elif config["model"] == "model0a":
                                model_to_fit = lambda t, waterlevel_p, tau_max, drop_eq, shift, scale:\
                                               func_sciopt(t,
                                               list_models=[func_rain, func_healing, func_temp1],
                                               list_vars=[[modelz, dp_rain, rhos, K_vs], [t], [t, temp_C, config["smoothing_temperature_n_samples"]]],
                                               list_params=[[waterlevel_p], [tau_max, drop_eq], [shift, scale]],
                                               n_channels=1)
                            elif config["model"] == "model2":
                                model_to_fit = lambda t, waterlevel_p, drop_eq, recovery, slope, const, shift, scale:\
                                       func_sciopt(t,
                                       list_models=[func_rain, func_quake, func_lin, func_temp1],
                                       list_vars=[[modelz, dp_rain, rhos, K_vs], [t], [t], [t, temp_C, config["smoothing_temperature_n_samples"]]],
                                       list_params=[[waterlevel_p], [drop_eq, recovery], [slope, const], [shift, scale]],
                                       n_channels=1)
                            elif config["model"] == "model3":
                                model_to_fit = lambda t, waterlevel_p, slope, const, shift, scale:\
                                       func_sciopt(t,
                                       list_models=[func_rain, func_lin, func_temp1],
                                       list_vars=[[modelz, dp_rain, rhos, K_vs], [t], [t, temp_C, config["smoothing_temperature_n_samples"]]],
                                       list_params=[[waterlevel_p], [slope, const], [shift, scale]],
                                       n_channels=1)
                            elif config["model"] == "model4":
                                model_to_fit = lambda t, phi, a, tau_max, drop_eq, slope, const, shift, scale:\
                                           func_sciopt(t,
                                           list_models=[func_pseudo_SSW, func_healing, func_lin, func_temp1],
                                           list_vars=[[t, rain_m], [t], [t], [t, temp_C, config["smoothing_temperature_n_samples"]]],
                                           list_params=[[phi, a], [tau_max, drop_eq], [slope, const], [shift, scale]],
                                           n_channels=1)
                            elif config["model"] == "model5":
                                model_to_fit = lambda t, factor_pp, tau_max, drop_eq, slope, const, shift, scale:\
                                           func_sciopt(t,
                                           list_models=[func_rain1, func_healing, func_lin, func_temp1],
                                           list_vars=[[modelz, dp_rain, rhos, K_vs, pressure_Pa], [t], [t], [t, temp_C, config["smoothing_temperature_n_samples"]]],
                                           list_params=[[factor_pp], [tau_max, drop_eq], [slope, const], [shift, scale]],
                                           n_channels=1)
                            elif config["model"] == "model6":
                                model_to_fit = lambda t, factor_pp, tau_max, drop_eq, shift, scale:\
                                           func_sciopt(t,
                                           list_models=[func_rain1, func_healing, func_temp1],
                                           list_vars=[[modelz, dp_rain, rhos, K_vs, pressure_Pa], [t], [t, temp_C, config["smoothing_temperature_n_samples"]]],
                                           list_params=[[factor_pp], [tau_max, drop_eq], [shift, scale]],
                                           n_channels=1)
                            else:
                                raise ValueError("Unknown model {} in config.".format(config["model"]))


                            bounds = set_bounds(config)

                            # Fitting
                            params_mod, covariance_mod =\
                                optimize.curve_fit(model_to_fit, t,
                                                   data_array.ravel(), sigma=sigma_array.ravel(),
                                                   bounds=bounds)

                            # save the arrays -- for convenience
                            foname = config["output_dir"] + "/data_{}.npy".format(datafob)
                            if not os.path.exists(foname):
                                np.save(foname, data_array[0])
                            foname = config["output_dir"] + "/timestamps_{}.npy".format(datafob)
                            if not os.path.exists(foname):
                                np.save(foname, t)
                            foname = config["output_dir"] + "/error_{}.npy".format(datafob)
                            if not os.path.exists(foname):
                                np.save(foname, sigma_array[0])
                            foname = config["output_dir"] + "/rain_m_{}.npy".format(datafob)
                            if not os.path.exists(foname):
                                np.save(foname, rain_m)
                            foname = config["output_dir"] + "/temp_C_{}.npy".format(datafob)
                            if not os.path.exists(foname):
                                np.save(foname, temp_C)


                            # set up the emcee sampler
                            if config["model"] in ["modelf", "modelfq", "model_rhophi"]:
                                indep_vars_emcee = [t, modelz, K_vs, K_vs_temp, rhos, phis, dp_rain, dp_temp, qtimes]
                            elif config["model"] in ["modelf_noquake"]:
                                indep_vars_emcee = [t, modelz, K_vs, K_vs_temp, rhos, phis, dp_rain, dp_temp]
                            elif config["model"] in ["model1", "model2", "model3"]:
                                indep_vars_emcee = [t, modelz, K_vs, rhos, dp_rain, temp_C, config["smoothing_temperature_n_samples"], config["time_resolution"]]
                            elif config["model"] in ["model0", "model0a"]:
                                indep_vars_emcee = [t, modelz, K_vs, rhos, dp_rain, temp_C, config["smoothing_temperature_n_samples"], config["time_resolution"], qtimes]
                            elif config["model"] == "model4":
                                indep_vars_emcee = [t, rain_m, temp_C, config["smoothing_temperature_n_samples"]]
                            elif config["model"] == "model5":
                                indep_vars_emcee = [t, modelz, K_vs, rhos, dp_rain, temp_C, pressure_Pa, config["smoothing_temperature_n_samples"], config["time_resolution"]]
                            elif config["model"] == "model6":
                                indep_vars_emcee = [t, modelz, K_vs, rhos, dp_rain, temp_C, pressure_Pa, config["smoothing_temperature_n_samples"], config["time_resolution"]]


                            # get the bounds
                            emcee_bounds = get_mcmc_bounds(config)

                            # get the initial position from the max. likeligood model and perturb by small random nrs
                            init_pos = get_initial_position_from_mlmodel(params_mod, config)

                            np.random.seed(42)
                            position = np.array([init_pos]*config["n_initializations"]) +\
                                np.random.randn(config["n_initializations"], len(init_pos)) *\
                                np.array(config["init_perturbation"][0: len(init_pos)])

                            nwalkers, ndim = (config["n_initializations"], len(init_pos))

                            with Pool(config["n_multiprocess"]) as pool:
                                # Initialize the sampler
                                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_for_emcee,
                                                                moves=[(emcee.moves.StretchMove(a=2.0), 0.5),
                                                                        (emcee.moves.DESnookerMove(), 0.5)],
                                                                args=(indep_vars_emcee, emcee_bounds, data_array, sigma_array, config["model"], config["use_logf"], config["use_g"]),
                                                                pool=pool)
                                iterations_performed = 0
                                while True:

                                    try:
                                        position = sampler.run_mcmc(position, config["n_iterations"], progress=True)
                                        iterations_performed += config["n_iterations"]
                                    except ValueError:  # badly conditioned starting point
                                        position = sampler.get_last_sample()
                                        position.coords += np.random.randn(config["n_initializations"], len(init_pos)) *\
                                                           np.array(config["init_perturbation"][0: len(init_pos)])
                                    try:
                                        tau = sampler.get_autocorr_time(discard=config["n_burnin"])
                                        foname = config["output_dir"] + "/tau_{}.npy".format(fob)
                                        np.save(foname, tau)
                                        thin = int(np.max(tau)) // 2
                                        print("Tau could be estimated, tau: ", np.max(tau))
                                        have_tau = True
                                        break
                                    except:
                                        print("Apparently no convergence yet, adding another {} samples.".format(config["n_iterations"]))
                                    if iterations_performed >= config["max_iterations"]:
                                        # give up
                                        break

                            if not have_tau:
                                thin = 2
                                #continue


                            # get and save the samples
                            fig, axes = plt.subplots(ndim, figsize=(10, 2*ndim), sharex=True)
                            all_samples_temp = sampler.get_chain(discard=config["n_burnin"])
                

                            # Plot the chains
                            labels = config["list_params"].copy()
                            ls = []
                            for l in labels:
                                if l in ["tau_max", "drop_eq", "recovery"]:
                                    for q in qtimes:
                                        ls.append(l)
                                else:
                                    ls.append(l)
                            labels = ls
                            if config["use_logf"]:
                                labels += ["log10_f"]
                            if config["use_g"]:
                                labels += ["g"]
                            for ixparam in range(ndim):
                                ax = axes[ixparam]
                                ax.plot(all_samples_temp[:, :, ixparam], "k", alpha=0.3)
                                ax.set_xlim(0, len(all_samples_temp))
                                ax.set_ylabel(labels[ixparam])
                                ax.yaxis.set_label_coords(-0.1, 0.5)
                            axes[-1].set_xlabel("step number")
                            foname = (config["output_dir"] + "/MCMC_chains_{}{}.png".format(fob, {True: "", False: "_noconv"}[have_tau]))
                            fig.savefig(foname)
                            plt.close()

                            
                            # Create a corner plot and save
                            flat_samples = sampler.get_chain(flat=True, discard=config["n_burnin"], thin=thin)
                            log_prob_samples = sampler.get_log_prob(discard=config["n_burnin"], flat=True, thin=thin)
                            if flat_samples.shape[0] > 50:
                                samples_probs = np.concatenate((flat_samples, log_prob_samples[:, None]), axis=1)
                            
                                labels += ["log prob"]
                                fig = corner.corner(
                                    samples_probs, labels=labels
                                );
                                foname = (config["output_dir"] + "/MCMC_{}{}.png".format(fob, {True: "", False: "_noconv"}[have_tau]))
                                fig.savefig(foname)
                                plt.close()

                            # save the "clean" ensemble: Post burn-in, flat, decimated by 1/2 * autocorrelation time.
                            foname = (config["output_dir"] + "/probability_{}{}.npy".format(fob, {True: "", False: "_noconv"}[have_tau]))
                            np.save(foname, log_prob_samples)
                            foname = (config["output_dir"] + "/samples_{}{}.npy".format(fob, {True: "", False: "_noconv"}[have_tau]))
                            np.save(foname, flat_samples)
                            
                            # get the median and percentile models and save
                            mcmcout = []
                            for ixp in range(ndim):
                                mcmcout.append(np.percentile(flat_samples[:, ixp], [16, 50, 84]))
                            mcmcout = np.array(mcmcout)
                            foname = (config["output_dir"] + "/percs_{}{}.npy".format(fob, {True: "", False: "_noconv"}[have_tau]))
                            np.save(foname, mcmcout)


                            # plot the data, max. likelihood, median, and best fit model
                            ixbest = np.argmax(log_prob_samples)
                            #ixbest = np.unravel_index(ixbest, prob[config["n_burnin"]:, :].shape)
                            maxprob_sample = flat_samples[ixbest, :]

                            if config["model"] == "model_rhophi":
                                dvv_mcmc = evaluate_model_rhophi(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "modelf":
                                dvv_mcmc = evaluate_modelf(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "modelf_noquake":
                                dvv_mcmc = evaluate_modelf_noquake(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "modelfq":
                                dvv_mcmc = evaluate_modelfq(indep_vars_emcee, maxprob_sample)
                            #elif config["model"] == "modelfa":
                            #    dvv_mcmc = evaluate_modelfa(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "model2":
                                dvv_mcmc = evaluate_model2(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "model0":
                                dvv_mcmc = evaluate_model0(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "model0a":
                                dvv_mcmc = evaluate_model0a(indep_vars_emcee, maxprob_sample)                           
                            elif config["model"] == "model1":
                                dvv_mcmc = evaluate_model1(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "model3":
                                dvv_mcmc = evaluate_model3(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "model4":
                                dvv_mcmc = evaluate_model4(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "model5":
                                dvv_mcmc = evaluate_model5(indep_vars_emcee, maxprob_sample)
                            elif config["model"] == "model6":
                                dvv_mcmc = evaluate_model6(indep_vars_emcee, maxprob_sample)



                            fig = plt.figure()
                            axa = plt.subplot(121)
                            plt.plot(t, data_array[0], "k")
                            plt.plot(t, model_to_fit(t, *params_mod), "seagreen", alpha=0.6)
                            plt.plot(t, dvv_mcmc, "rebeccapurple")
                            plt.grid()
                            plt.ylim(data_array[0].min() * 0.9, data_array[0].max() * 1.1)
                            plt.xlabel("timestamps (s)")
                            plt.legend(["data", "model max. l", "model max. a p"], loc=3, fontsize="small")

                            axb = plt.subplot(122)
                            plt.plot(t, (data_array[0] - dvv_mcmc), "purple")
                            plt.grid()
                            plt.xlabel("timestamps (s)")
                            plt.legend(["Residual"], loc=3, fontsize="small")
                            plt.tight_layout()
                            plt.savefig(config["output_dir"] + "/testplot2{}{}.png".format(fob, {True: "", False: "_noconv"}[have_tau]))

                            # print info
                            print("Median neg. log. probability for MC inversion: ",
                                  log_probability_for_emcee(mcmcout[:, 1], indep_vars_emcee, emcee_bounds, data_array, sigma_array, config["model"], config["use_logf"], config["use_g"]))
                            print("Model params: ", flat_samples[ixbest, :])

                            plt.close() 
