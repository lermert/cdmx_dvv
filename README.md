### Data analysis for dv/v measurements for manuscript (title tbd)

This set of scripts can be used to run Monte Carlo inversions with the emcee module (https://github.com/dfm/emcee) so as to obtain optimal-fit parameters for modeling velocity change in Mexico City. 

To reproduce our inversions:
1. Set up a conda environment following the environment yaml file in the reproducibility directory (to be added)
2. Head to zenodo to obtain the data files (to be added), which contain time stamps, dv/v measurements and various other input.
3. Select the relevant input yaml file from the reproducibility folder and edit it to your liking.
4. Run: python m.py \<path-to-input.yml\>


Brief provenance of the data in the zenodo repository: Correlations were obtained with ants https://github.com/lermert/ants_2, using the input files (with minor adaptations for different data sources, instruments etc) in reproducibility folder (to be added). Clustering, stacking and stretching measurements were performed using the ruido module https://github.com/lermert/ruido, again with input files as in reproducibility folder. 

