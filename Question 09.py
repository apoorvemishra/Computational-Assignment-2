import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.integrate import solve_ivp as ivp
from scipy.integrate import solve_bvp as bvp

#Question 9
print("Question 9 solutions")

#1st equation
def ode_system(x, y):
    dydx = np.zeros((2, x.size))
    dydx[0] = y[1]  # y[1] = dy/dx
    dydx[1] = -np.exp(-2*y[0])  # y'' = -exp(-2y)
    return dydx

def boundary_conditions(ya, yb):
    # Boundary conditions: y(1) = 0, y(2) = ln(2)
    return np.array([ya[0], yb[0] - np.log(2)])

# Initial guess for the solution
x_guess = np.linspace(1, 2, 5)
y_guess = np.zeros((2, x_guess.size))

# Solve the boundary value problem
sol = bvp(ode_system, boundary_conditions, x_guess, y_guess)

# Generate points for plotting
x_plot = np.linspace(1, 2, 100)
y_plot = sol.sol(x_plot)[0]
true_solution = lambda x_plot: np.log(x_plot)

# Plot the solution
plt.plot(x_plot, y_plot, label='Numerical Solution')
plt.plot(x_plot, true_solution(x_plot), label='Analytical Solution')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Question 9 ODE y\'\' = -exp(-2y) using solve_bvp')
plt.legend()
plt.grid(True)
plt.figure()


#2nd equation
def ode_system(x, y):
    dydx = np.zeros((2, x.size))
    dydx[0] = y[1]  # y[1] = dy/dx
    dydx[1] = y[1] * np.cos(x) - y[0] * np.log(y[0])  # y'' = y'*cos(x) - y*ln(y)
    return dydx

def boundary_conditions(ya, yb):
    # Boundary conditions: y(0) = 1, y(pi/2) = e
    return np.array([ya[0] - 1, yb[0] - np.exp(1)])

# Initial guess for the solution
x_guess = np.linspace(0, np.pi/2, 10)
y_guess = np.zeros((2, x_guess.size))
y_guess[0] = np.linspace(1, np.exp(1), x_guess.size)

# Solve the boundary value problem
sol = bvp(ode_system, boundary_conditions, x_guess, y_guess)

# Generate points for plotting
x_plot = np.linspace(0, np.pi/2, 100)
y_plot = sol.sol(x_plot)[0]
true_solution = lambda x_plot: np.exp(np.sin(x_plot))

# Plot the solution
plt.plot(x_plot, y_plot, label='Numerical Solution')
plt.plot(x_plot, true_solution(x_plot), label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Question 9 ODE y\'\' = y\'*cos(x) - y*ln(y) using solve_bvp')
plt.legend()
plt.grid(True)
plt.figure()


#3rd equation
def ode_system(x, y):
    dydx = np.zeros((2, x.size))
    dydx[0] = y[1]  # y[1] = dy/dx
    dydx[1] = -(2 * (y[1])**3 + y[0]**2 * y[1]) / np.cos(x)  # y'' = -(2*(y')^3 + y^2*y') / sec(x)
    return dydx

def boundary_conditions(ya, yb):
    # Boundary conditions: y(pi/4) = 2**(-1/4), y(pi/3) = 12**0.25 / 2
    return np.array([ya[0] - 2**(-1/4), yb[0] - (12**0.25) / 2])

# Initial guess for the solution
x_guess = np.linspace(np.pi/4, np.pi/3, 10)
y_guess = np.zeros((2, x_guess.size))
y_guess[0] = np.linspace(2**(-1/4), (12**0.25) / 2, x_guess.size)

# Solve the boundary value problem
sol =bvp(ode_system, boundary_conditions, x_guess, y_guess)

# Generate points for plotting
x_plot = np.linspace(np.pi/4, np.pi/3, 100)
y_plot = sol.sol(x_plot)[0]
true_solution = lambda x_plot: np.sin(x_plot)**0.5

# Plot the solution
plt.plot(x_plot, y_plot, label='Numerical Solution')
plt.plot(x_plot, true_solution(x_plot), label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Question 9 ODE y\'\' = -(2*(y\')^3 + y^2*y\') / sec(x) using solve_bvp')
plt.legend()
plt.grid(True)
plt.figure()


#4th equation
def ode_system(x, y):
    dydx = np.zeros((2, x.size))
    dydx[0] = y[1]  # y[1] = dy/dx
    dydx[1] = 0.5 - (y[1]**2)/2 - y[0] * np.sin(x)/2  # y'' = 1/2 - (y')^2/2 - y*sin(x)/2
    return dydx

def boundary_conditions(ya, yb):
    # Boundary conditions: y(0) = 2, y(pi) = 2
    return np.array([ya[0] - 2, yb[0] - 2])

# Initial guess for the solution
x_guess = np.linspace(0, np.pi, 100)
y_guess = np.zeros((2, x_guess.size))

# Solve the boundary value problem
sol = bvp(ode_system, boundary_conditions, x_guess, y_guess)

# Generate points for plotting
x_plot = np.linspace(0, np.pi, 100)
y_plot = sol.sol(x_plot)[0]
true_solution = lambda x_plot: 2 + np.sin(x_plot)

# Plot the solution
plt.plot(x_plot, y_plot, label='Numerical Solution')
plt.plot(x_plot, true_solution(x_plot), label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Question 9 ODE y\'\' = 1/2 - (y\')^2/2 - y*sin(x)/2 using solve_bvp')
plt.legend()
plt.grid(True)
plt.show()



print('***************************************************')
