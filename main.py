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

#rejection sampling

M_CONST = target_pdf(0)

def sample_rejection(n_samples_needed):

    samples = []
    n_attempts = 0 #tracker 

    
    # The expected number of attempts is n_samples_needed / (1/M)
    n_estimate = int(n_samples_needed * M_CONST * 1.5) # Overestimate attempts

    while len(samples) < n_samples_needed:
        # large batch for efficiency (vectorization)
        batch_size = min(n_samples_needed * 2, 100000)
        
        # propose x* from proposal g(x) = U(0, 1)
        x_star = np.random.uniform(0, 1, batch_size)
        
        # evaluate f(x*)
        f_x_star = target_pdf(x_star)
        
        # acceptance test: u <= f(x*) / (M * g(x*))
        # g(x*) = 1, so the test is u <= f(x*) / M_CONST
        u = np.random.uniform(0, 1, batch_size)
        
        # finds the accepted samples
        accepted_mask = u <= f_x_star / M_CONST
        accepted_samples = x_star[accepted_mask]
        samples.extend(accepted_samples.tolist())
        n_attempts += batch_size

    # calculate efficiency
    final_samples = np.array(samples[:n_samples_needed])
    n_accepted = len(final_samples)
    efficiency = n_accepted / n_attempts
    
    return final_samples, efficiency