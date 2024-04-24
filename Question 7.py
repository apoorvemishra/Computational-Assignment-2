import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 7
print("Question 7 solutions")
h = 0.1
t = np.arange(0,10+h,h)
w = np.zeros(len(t))
tol = 10e-4

for j in range(10**5):
    w_last = w.copy()
    for i in range(len(t)-1):
        w[i] = 0.5*(w[i-1]+w[i+1]) + 0.5*10*h**2
    if (j+1)%1000==0:
        plt.plot(t,w, label = f"{j+1}th iteration")
    if max(abs(w-w_last))<tol: break
plt.plot(t,w, label = f"{j+1}th iteration")

def x(t):
    return 50*t - 5*t*t
plt.plot(t,x(t), label = "orignal solution")
plt.title("Question 7 Relaxation Method")
plt.legend()
plt.show()

print('***************************************************\n')
