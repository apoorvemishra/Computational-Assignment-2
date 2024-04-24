import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 6
print("Question 6 solutions")
h = 0.01
t = np.arange(0,10+h,h)
w = np.zeros(len(t))
wp = np.zeros(len(t))
w[0] = 0
w[-1] = 1
tol = 10**(-4)
iteration = 1

while abs(w[-1])>tol:
    for i in range(len(t)-1):
        wp[i+1] = wp[i] - h*10
        w[i+1] = w[i] + h*wp[i]
    plt.plot(t,w, label = f"{iteration}th iteration")
    iteration = iteration + 1
    wp[0] = wp[0] - w[-1]/9     #updation step for initial slope
    if iteration>20: break

def x(t):
    return 50*t - 5*t*t
plt.plot(t,x(t), label = "orignal solution")
plt.title("Question 6 Shooting method")
plt.legend()
plt.show()

print('***************************************************\n')
