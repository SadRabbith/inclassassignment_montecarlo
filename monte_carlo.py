import numpy as np
import matplotlib.pyplot as plt
import random

def ellipse(x, a=5,b=2):
    sol =  np.sqrt(b**2*(1-x**2/a**2))
    return sol


#We see that we should uniformly sample points from the 

def integrate(f, N):
    hits = 0 #number of points inside area
    for i in range(N):
        x = 5*random.random() #random number between 0 and 5
        y = 2*random.random() #random number between 0 and 2

        if y<f(x):
            hits+=1   #if random point is under function, iterate hits

    integral = (hits/N)*10  #integral is probability of hit * total area sampled

    return 4*integral #4x, since we're sampling from only the first quadrant
print("Computed Value: ",integrate(ellipse,10000))
print("Analytic Value: ",np.pi*5*2)


N = np.arange(1,10000,100) #
int_val = np.zeros(len(N))

analytic_val = np.pi * 5 * 2
for i in range(len(N)):
    int_val[i] += integrate(ellipse,N[i])

plt.plot(N, np.abs(int_val-analytic_val)/analytic_val)
plt.plot(N, 1/np.sqrt(N),label=r"$1/\sqrt{N}$",linestyle='dashed',color='red')
plt.xlabel(r"$N$", fontsize=14)
plt.legend(fontsize=14)
plt.ylabel("Relative Error", fontsize=14)
plt.title("Error Scaling with Number of Samples",fontsize=14)
plt.savefig("integral_error_scaling.png", dpi=300)
plt.clf()

################## FINDING CIRCUMFERENCE #################
epsilon = 0.1
ellipse_2 = lambda x: ellipse(x, 5-epsilon, 2-epsilon)
x = np.linspace(0,5,1000)
plt.plot(x,ellipse(x))
plt.plot(x,ellipse_2(x))
plt.show()
def circumference_estimate(f1,f2, N):
    hits = 0 #number of points inside area
    for i in range(N):
        x = 5*random.random() #random number between 0 and 5
        y = 2*random.random() #random number between 0 and 2

        if y<f1(x) and y>f2(x):
            hits+=1   #if random point is under function, iterate hits

    integral = (hits/N)*10  #integral is probability of hit * total area sampled

    return 4*integral

numerical_circumference = circumference_estimate(ellipse, ellipse_2, 10000000)
print(numerical_circumference)
