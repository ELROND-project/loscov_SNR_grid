# loscov SNR grid
Supplementary (though still self-contained) code to https://github.com/ELROND-project/loscov, used to test the observability of the auto and cross-correlations of the line-of-sight shear of strong gravitational lenses L with galaxy shapes E and galaxy positions P, used in Fleury et al. 2026.

This current version is a work in progress, and may be updated in future with general speed-ups and other improvements.

Authors: Daniel Johnson, Pierre Fleury, Theo Duboscq, Natalie Hogg

## Running the code

To generate the relevant correlation functions and covariance matrices,

0. set run parameters in config.py - here you can specify survey properties, cosmology, numerical parameters etc

1. run part_one.sh, which processes the redshift distributions, calculates the relevant correlation functions and defines the grid of sigma_L (the measurement uncertainty in the shear) and Nlens (the number of lenses) values for which the signal-to-noise will be computed

2. run part_two.sh. This calculates and saves interpolated versions of functions within the various covariance matrices, which will later be used to efficiently optimise the SNR as a function of theta. This also computes a smoothed version of the cosmic covariance, to reduce the impact of noise from the monte carlo integration

After completing step 2, you can use testing_SNR_optimisation.ipynb to view the interpolated functions and the corresponding SNR vs theta plot for a given sigma_L and Nlens, to see that the level of smoothing is sufficient.

3. run part_three.sh. This determines a value of theta such that the SNR in an angular bin [0,theta] is maximised for a given correlation function, sigma_L and Nlens, and saves the corresponding signal, variance, SNR and variance components (noise, sparsity and cosmic)

4. run part_four.sh. This simply combines the many resulting data files into a single file, which makes subsequent processing faster and easier

To redo the smoothing without needing to repeat steps 1 and 2, run re-smoothing.sh

To redo the generation of the sigma_L and Nlens grid without needing to repeat steps 1 and 2, run generate_params.sh


The resulting data is saved in within the data folder. The SNR grid figures are plotted using plotter.ipynb. The correlation functions and redshift distributions are plotted with plotting_correlations.ipynb, and the testing_SNR_optimisation.ipynb can be used to examine the SNR as a function of theta for an observable of interest

Note that the .sh files are adapted to the particular computing cluster used by the authors, and should be adapted (as well as the relevant parameters in the config.py file) 

## Attribution

Please cite Fleury et al. 2026 if loscov is used for a publication, and link https://github.com/ELROND-project/loscov_SNR_grid. Feel free to get in touch with any questions you might have!
