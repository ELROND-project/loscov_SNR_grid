import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import *


############################################ USEFUL FUNCTIONS ###########################################

### converting angles

def radtoarcmin(angle_rad):
    """
    This function converts an an angle expressed in radians
    into arcmins.
    """
    
    angle_arcmin = angle_rad * 60 * 180 / np.pi
    
    return angle_arcmin


def arcmintorad(angle_arcmin):
    """
    This function converts an an angle expressed in arcmins
    into radians.
    """
    
    angle_rad = angle_arcmin / (60 * 180 / np.pi)
    
    return angle_rad

### delta function

def delta_func(a,b):
    if a == b:
        x = 1
    else:
        x = 0

    return x

def roundsf(x, sig_figs=1):
    if x == 0:
        return 0  # Avoid log10 issues
    power = math.floor(math.log10(abs(x)))  # Find order of magnitude
    factor = 10 ** power  # Get the scaling factor
    return round(x / factor) * factor  # Round and scale back

## saving data in a pickle file

def save_pickle(data, filename, descriptor):
    """Helper function to save data to a pickle file."""
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {descriptor} at {filename}")
    except Exception as ex:
        print(f"Error during pickling {descriptor}: {ex}")

def add_dict(*objects):
    """Automatically adds multiple objects to global_dict with their variable names as keys."""
    frame = inspect.currentframe().f_back
    for name, value in frame.f_locals.items():
        if any(value is obj for obj in objects):  # Use identity check
            global_dict[name] = value  # Add it to global_dict

def get_item(*names):
    """Retrieves items from global_dict and defines them as global variables in the caller's module."""
    frame = inspect.currentframe().f_back  # Get caller's frame
    caller_globals = frame.f_globals  # Access the caller's global namespace
    
    for name in names:
        if name in global_dict:
            caller_globals[name] = global_dict[name]  # Define the variable globally
        else:
            raise KeyError(f"'{name}' not found in global_dict")

def load_correlations(filename="correlations.pkl"):
    """Loads a pickled dictionary from 'filename' and adds its contents to global_dict."""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)  # Load the dictionary from the pickle file
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
    except pickle.UnpicklingError:
        raise ValueError(f"File '{filename}' is not a valid pickle file.")

    if isinstance(data, dict):
        global_dict.update(data)  # Merge the loaded dictionary into global_dict
    else:
        raise ValueError("The pickled file does not contain a dictionary.")

def load_file(filename):
    """Loads a pickled dictionary"""
    with open(filename, "rb") as f:
        data = pickle.load(f)  # Load the dictionary from the pickle file

    return(data)

###################################### Monte Carlo Integrator #######################################

def monte_carlo_integrate(func, bounds, num_samples=nsamp, confidence=confidence):
    """
    Monte Carlo integration over a given domain with error estimation.
    
    Parameters:
    - func (callable): The function to integrate. NB it needs to accept an array as input
                       for integration over multiple dimensions.
    - bounds (list of tuples): Integration bounds [(a1, b1), (a2, b2), ...].
                                For 1D, use [(a, b)].
    - num_samples (int): Number of random samples to use to achieve a 10% error (adjusted via desired_precision)
    - confidence (float): Confidence level for the error estimate (default is 0.95) (this is something ChatGPT
                             recommended when I was figuring out how to estimate the error)
    
    Returns:
    - tuple: (float, float) Estimated value of the integral and its error.
    """

    nsamp_use = num_samples * (10/desired_error)**2

    nsamp_use = int(max(nsamp_use,minsamp))    #don't use fewer samples than the minimum

    nsamp_use = int(min(nsamp_use,maxsamp))    #don't use more samples than the maximum
    
    # Determine the dimensionality and the volume of the integration domain
    dim = len(bounds)
    volumes = [b - a for a, b in bounds]
    total_volume = np.prod(volumes)
    rng = np.random.default_rng()
    
    # Generate random samples within the bounds
    samples = np.array([rng.uniform(low=a, high=b, size=nsamp_use) for a, b in bounds])
    
    # Evaluate the function at the random sample points
    values = func(samples)
    
    # Compute the Monte Carlo estimate of the integral
    integral = total_volume * np.mean(values)
    
    # Estimate the standard error
    if values.size > 1:
        variance = np.var(values, ddof=1)  # Sample variance
    else:
        variance = 0
        print("Error! Too few values")
    std_error = total_volume * np.sqrt(variance / nsamp_use)
    
    # Compute the confidence interval using the normal distribution
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)  # z-score for the confidence level
    error = z * std_error
    # if integral != 0:
        # print("Integration error = " + str(int(100 * np.abs(error/integral))) + "%")
        # if np.abs(error/integral) > warning_level:
            # print("WARNING: integration error = " + str(int(100 * np.abs(error/integral))) + "%")
    
    return integral, error