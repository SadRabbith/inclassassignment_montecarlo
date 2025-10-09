import numpy as np 
import matplotlib.pyplot as plt


from scipy.stats import expon 

# our function = exp(-x) / 1 - exp(-1)

C_NORM = 1 / (1 - np.exp(-1))

def target_pdf(x):
    
    # Use np.where to ensure the function exists only in the [0, 1] range
    return np.where((x >= 0) & (x <= 1), C_NORM * np.exp(-x), 0)

#  inverse CDF -> -ln(1 - u * (1 - e^-1))
def inverse_cdf(u):
   
    return -np.log(1 - u * (1 - np.exp(-1)))

#now the sampling, generating uniform random numbers u in U(0,1) 

def sample_inverse_cdf(n_samples):
    u_samples = np.random.uniform(0, 1, n_samples)    
    samples = inverse_cdf(u_samples)
    return samples

