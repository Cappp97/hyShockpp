# IMPORT LIBRARIES
import mutationpp as mpp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math as mt

print('Succesfully imported shared libraries')

# PRESHOCK CONDITIONS
M_inf = 28.6  # freestream mach number
T_inf = 300  # freestream temperature [K]
P_inf = 13.33  # freestream pressure [Pa]

# MIXTURE
opts = mpp.MixtureOptions("air_11_omar")
opts.setStateModel("ChemNonEq1T")
opts.setThermodynamicDatabase("RRHO")
mix = mpp.Mixture(opts)

R = 8.31446261815324  # UNIVERSAL GAS CONSTANT
Ri = [R / mix.speciesMw(i) for i in range(mix.nSpecies())]  # SPECIES GAS CONSTANT
print('Succesfully initialized required mixture')

# EQUILIBRIUM CONDTIONS
mix.equilibrate(T_inf, P_inf)  # set the mixture in chemical equilibrium before the shock
rho_inf = mix.density()  # freestream density [kg/m^3]
c_inf = mix.equilibriumSoundSpeed()  # sound speed
u_inf = M_inf * mix.equilibriumSoundSpeed()  # air speed
gamma = mix.mixtureEquilibriumGamma()  # specific heat ratio
h_inf = mix.mixtureHMass()  # enthalpy
Yi = mix.Y()  # mass ratios
print('Initial conditions computed')

# SUITABLE FIRST GUESS FOR THE POST SHOCK STATE
# a suitable post shock state can be found from the Rankine Hugonit jump conditions for a politropic ideal std. air mixture
P_0 = ((2 * gamma / (gamma + 1)) * (M_inf ** 2 - 1) + 1) * P_inf
rho_0 = rho_inf * (1 - (2 / (gamma + 1)) * (1 - (1 / M_inf ** 2))) ** (-1)
u_0 = u_inf - (2 * c_inf / (gamma + 1)) * (M_inf - (1 / M_inf))
T_0 = P_0 / (rho_0 * 287)


# NONLINEAR SOLVER FOR THE RANKINE HUGONIOT JUMP CONDITIONS RK JUMP The state vector are the variables u, T, P (Note,
# any other set of variables can be used, these are not the conserved variables) Returns the residual of the RK jump
# condtion at a given state The nonlinear solver will try to set the residual to zero
def rk_jump(state):
    global u_inf, rho_inf, h_inf, mix, Yi  # global variables

    u, T, P = state  # current state

    # This piece of code is necessary to avoid unphysical conditions therefore avoiding stability and convergence issues
    # If the solver ends up in an unphyscal state the code immediatly returns a very high value
    err_value = 1e+17
    if P < 0 or T < 0 or u < 0:
        return [err_value, err_value, err_value]

    # Set the current mixture state
    # because we are using frozen chemistry, the concentrations will be fixed
    # Note: Mixture is not in chemical equilibrium
    mix.setState(Yi, [P, T], 2)

    # get conserved variables
    rho = mix.density()
    h = mix.mixtureHMass()

    # Rankine Hugoniot jump conditions
    eq1 = u_inf * rho_inf - rho * u
    eq2 = P_inf + rho_inf * u_inf ** 2 - P - rho * u ** 2
    eq3 = h_inf + 0.5 * u_inf ** 2 - h - 0.5 * u ** 2
    return [eq1, eq2, eq3]


# Frozen chemistry solution
print('Solving the frozen chemistry shock model')
solution = fsolve(rk_jump, np.array([u_0, T_0, P_0]))
err = np.linalg.norm(rk_jump(solution), 2)  # L2 norm of the residual
init_to_print = np.round([u_inf, T_inf, P_inf], 2)
sol_to_print = np.round(solution, 2)
const_to_print = np.round([u_0, T_0, P_0], 2)

print('''
NONLINEAR SHOCK SOLVER SOLUTION
INITIAL CONDITION            SOLUTION               CONSTANT CP SOLUTION
U = {0} [m/s]               U = {3} [m/s]           U = {6} [m/S]            
T = {1} [K]                 T = {4} [K]             T = {7} [K]
P = {2} [Pa]                P = {5} [Pa]            P = {8} [Pa]
ERROR
eps = {9}
'''.format(*init_to_print, *sol_to_print, *const_to_print, err))

# SHOCK RELAXATION
# Chosen variables are rho, u, e, Yi

state = np.array([mix.density(), solution[0], mix.mixtureEnergyMass(), *Yi])


# D_STATE Returs the derivative of the state vector by using the Euler Equations.
# Note, Euler Equations are written in
# the form J*dx = b where dx is the state derivative, J the jacobian and b the source term The code needs to solve
# the previous system to get the state derivative

def d_state(x, state):
    rho = state[0]
    u = state[1]
    e = state[2]
    Ys = state[3:]

    # Set the current mixture state, note: as state variables we use, for pure convenience, the densities rho_i and the
    # internal energy (any other combination accepted by mutation would have worked
    mix.setState(rho * Ys, rho * e, 0)

    # GET IMPORTANT QUANTITIES TO COMPUTE THE A MATRIX
    P = mix.P()
    T = mix.T()

    Cv = mix.mixtureFrozenCvMass()
    wi = mix.netProductionRates()

    #THERMODYNAMIC DERIVATIVES
    dP_dYi = [rho * Ri[i] * T for i in range(mix.nSpecies())]  # derivative of pressure wrt species concentration
    dP_dRho = sum([Ys[i] * Ri[i] * T for i in range(mix.nSpecies())]) #derivative of pressure wrt to density
    dP_dT = rho*sum([Ys[i] * Ri[i]  for i in range(mix.nSpecies())]) #derivative of pressure wrt energy

    # COMPUTE EACH ROW OF THE A MATRIX
    row1 = np.hstack([[u, rho, 0], np.zeros_like(Ys)])
    row2 = np.hstack([[dP_dRho, rho * u, dP_dT/Cv], dP_dYi])
    row3 = np.hstack([[-P, 0, rho * rho], np.zeros_like(Ys)])

    diagYs = np.diag(np.multiply(rho * u, np.ones_like(Ys)))
    last_rows = np.hstack([np.zeros((mix.nSpecies(), 3)), diagYs])

    A = np.vstack([row1, row2, row3, last_rows])
    # COMPUTE THE SOURCE TERM
    b = np.hstack([[0, 0, 0], wi])
    print(mix.P() + rho * u * u)

    # SOLVE THE SYSTEM AND RETURN THE SOLUTION
    return np.linalg.solve(A, b)


x = (0, 0.1)
print('Integrating system')
sol = solve_ivp(d_state, x, state, t_eval=np.linspace(x[0], x[1], 1000), method="LSODA")

import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0, :])
plt.show()
print(mix.T(), mix.P())
