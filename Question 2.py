
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 2
print("Question 2 solutions")
h = 0.1
t = np.arange(1,2,h)
w = np.zeros(int(1/h))
w[0] = 1
for i in range(len(t)-1):
    w[i+1] = w[i] + h*(1-w[i]/t[i])*(w[i]/t[i])


plt.plot(t,w, label = 'Euler\'s method solution')

def y(t):
    return t/(1 + np.log(t))
plt.plot(t,y(t), label = 'Analytical solution')

plt.legend()
plt.title("Question 2")
plt.show()


# Error analysis
err, absErr = [],[]
for i in range(len(t)):
    g = y(t[i]) - w[i]
    err.append(g)
    absErr.append(g/y(t[i]))

print('Maximum of Absolute error was %f and maximum of relative erroe was %f' % (max(err), max(absErr)))
print('***************************************************\n')
