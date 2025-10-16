import numpy as np
import matplotlib.pyplot as plt
import random

a = 5
b=2
def ellipse(x, a=5, b=2):
    val = 1 - (x**2) / (a**2)
    return np.sqrt(b**2 * np.maximum(val, 0.0))

def integrate(f, N):
    hits = 0 #number of points inside area
    for i in range(N):
        x = a*random.random() #random number between 0 and 5
        y = b*random.random() #random number between 0 and 2

        if y<f(x):
            hits+=1   #if random point is under function, iterate hits

    integral = (hits/N)*10  #integral is probability of hit * total area sampled

    return 4*integral #4x, since we're sampling from only the first quadrant
print("Area Computed Value: ",integrate(ellipse,10000))
print("Area Analytic Value: ",np.pi*a*b)

# N = np.arange(1,10000,100) 
# int_val = np.zeros(len(N))

# analytic_val = np.pi * 5 * 2
# for i in range(len(N)):
#     int_val[i] += integrate(ellipse,N[i])

# plt.plot(N, np.abs(int_val-analytic_val)/analytic_val)
# plt.plot(N, 1/np.sqrt(N),label=r"$1/\sqrt{N}$",linestyle='dashed',color='red')
# plt.xlabel(r"$N$", fontsize=14)
# plt.legend(fontsize=14)
# plt.ylabel("Relative Error", fontsize=14)
# plt.title("Error Scaling with Number of Samples",fontsize=14)
# plt.savefig("integral_error_scaling.png", dpi=300)
# plt.clf()

################## FINDING CIRCUMFERENCE #################
epsilon = 0.1
ellipse_2 = lambda x: ellipse(x, a-epsilon, b-epsilon)

def circumference_estimate(f1,f2, N):
    hits = 0 #number of points inside area
    for i in range(N):
        x = a*random.random() #random number between 0 and 5
        y = b*random.random() #random number between 0 and 2

        if y<f1(x) and y>f2(x):
            hits+=1   #if random point is under function, iterate hits

    area_shell = 4 * (hits / N) * (a * b) # area / epsilon is circumference
    return area_shell / epsilon

h = ((a - b)**2) / ((a + b)**2)
analytic_circ = np.pi * (a + b) * (1 + (3*h)/(10 + np.sqrt(4 - 3*h)))

print("Circumference Computed Value: ", circumference_estimate(ellipse, ellipse_2, 100000))
print("Circumference Analytic Value: ", analytic_circ)

## plotting statistical uncertainty vs numpber of samples
def mc_stats(estimator, n_samples, n_trials=50):
    # find standard deviation(statistical uncertainty)
    estimates = np.array([estimator(n_samples) for _ in range(n_trials)])
    return np.std(estimates, ddof=1)

Ns = np.arange(100,5000,100)
area_uncert = []
circ_uncert = []

for N in Ns:
    area_uncert.append(mc_stats(lambda n: integrate(ellipse, n), N))
    circ_uncert.append(mc_stats(lambda n: circumference_estimate(ellipse, ellipse_2, n), N))

plt.figure(figsize=(10,4))
plt.loglog(Ns, area_uncert, 'o-', label='Area uncertainty')
plt.loglog(Ns, circ_uncert, 's-', label='Circumference uncertainty')
# here its 1/N^1/2 because monte carlo gets more accurate with more N
plt.loglog(Ns, area_uncert[0]*(Ns[0]**0.5)/(Ns**0.5), 'k--', label='1/N^(1/2)')
plt.xlabel('Number of samples N')
plt.ylabel('Standard deviation')
plt.title('Statistical Uncertainty')
plt.legend()
plt.grid(True, which='both')
plt.savefig('uncertainty_scaling.png', dpi=300)
plt.show()

# to improve efficiency, sample only near the ellipse for circumference (importance region)

def circumference_importance(N):
    # only samples near shell
    xs = np.random.rand(N) * a              
    heights = ellipse(xs) - ellipse_2(xs)           # shell height at each x
    area_shell_est = 4 * a * np.mean(heights)       # monte carlo
    return area_shell_est / epsilon     

circ_uncert_importance = [mc_stats(lambda n: circumference_importance(n), N) for N in Ns]

plt.figure(figsize=(8,4))
plt.loglog(Ns, circ_uncert, 'o-', label='Uniform shell, same as before')
plt.loglog(Ns, circ_uncert_importance, 's-', label='Chosen Sampling, near ellipse')
plt.loglog(Ns, circ_uncert[0]*(Ns[0]**0.5)/(Ns**0.5), 'k--', label='~1/sqrt(N)')
plt.xlabel('Number of samples N')
plt.ylabel('Uncertainty')
plt.title('Improved efficiency for circumference calc')
plt.legend()
plt.grid(True, which='both')
plt.savefig('circ_variance_reduction.png', dpi=300)
plt.show()
