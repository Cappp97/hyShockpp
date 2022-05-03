import mutationpp as mpp
import numpy as np
import math as mt
from scipy.optimize import fsolve

# PRESHOCK CONDITIONS
M_inf = 28.68  # freestream mach number
T_inf = 300  # freestream temperature [K]
P_inf = 13.33  # freestream pressure [Pa]
rho_inf = 1.612e-4  # freestream density [kg/m^3]

# CREATE THE MIXTURE
opts = mpp.MixtureOptions("air_11")
opts.setStateModel("ChemNonEq1T")
mix = mpp.Mixture(opts)
mix.equilibrate(T_inf, P_inf)
c_inf = mix.equilibriumSoundSpeed()
u_inf = M_inf * mix.equilibriumSoundSpeed()  # equilibrium speed
gamma = mix.mixtureEquilibriumGamma()
h_inf = mix.mixtureHMass()
Yi = mix.Y()

# GET SUITABLE FIRST GUESS FOR THE POST SHOCK STATE
# a suitable post shock state can be found from the Rankine Hugonit jump conditions for a politropic ideal gas
P_2 = ((2 * gamma / (gamma + 1)) * (M_inf ** 2 - 1) + 1) * P_inf
rho_2 = rho_inf * (1 - (2 / (gamma + 1)) * (1 - (1 / M_inf ** 2))) ** (-1)
u_2 = u_inf - (2 * c_inf / (gamma + 1)) * (M_inf - (1 / M_inf))
T_2 = P_2 / (rho_2 * 287)


def rk_jump(state):
    global u_inf, rho_inf, h_inf, mix, Yi

    u, T, P = state

    if P < 0 or T < 0 or u < 0:
        return [1e+16, 1e+16, 1e+16]

    mix.setState(Yi, [P, T], 2)
    rho = mix.density()

    h = mix.mixtureHMass()
    eq1 = u_inf * rho_inf - rho * u
    eq2 = P_inf + rho_inf * u_inf ** 2 - P - rho * u ** 2
    eq3 = h_inf + 0.5 * u_inf ** 2 - h - 0.5 * u ** 2
    return [eq1, eq2, eq3]


solution = fsolve(rk_jump, np.array([u_2, T_2, P_2]))

print('''
INITIAL SOLUTION            CONSTANT CP SOLUTION
U = {0} [m/s]   U = {3} [m/s]
T = {1} [K]     T = {4} [K]
P = {2} [Pa]    P = {5} [Pa]

SOLUTION
U = {6} [m/s]
T = {7} [K]
P = {8} [Pa]
'''.format(u_inf, T_inf, P_inf, u_2, T_2, P_2, *solution))
