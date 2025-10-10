import numpy as np 
import matplotlib.pyplot as plt

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

# graphing to compare the distributions
N = 5000 # some large sampling
samples_inv = sample_inverse_cdf(N)
samples_rej, eff = sample_rejection(N)

x_grid = np.linspace(0, 1, 200)
pdf_vals = target_pdf(x_grid)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, samples, title in zip(
    axes,
    [samples_inv, samples_rej],
    ["Inverse CDF", f"Rejection (eff={eff:.2f})"]
):
    ax.hist(samples, bins=50, density=True, alpha=0.6, label="samples")
    ax.plot(x_grid, pdf_vals, "r-", lw=2, label="theoretical pdf")
    ax.set_xlim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()