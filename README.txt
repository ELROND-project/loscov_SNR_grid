0. Change variables within the config.py file according to your preference

1. run part_one.sh. This defines the correlation functions if they haven't yet been calculated, the redshift distributions, and creates the params.txt and tasks.txt files

2. run part_two.sh. This does all the interpolations we need to later optimise our snr

After part 2, it's worth playing around with testing_SNR_optimisation.ipynb to see that you're happy with the smoothing

3. run part_three.sh. This does the optimising and saves the resulting covariance values

4. run part_four.sh. This combines all the data into a single pickled file (easier to manipulate)

To redo the smoothing without needing to repeat steps 1 and 2, run re-smoothing.sh

To redo the generation of the sigma_L and Nlens grid without needing to repeat steps 1 and 2, run generate_params.sh