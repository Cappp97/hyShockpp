import mutationpp as mpp
import numpy as np
import math as mt
from scipy.optimize import fsolve

CaseData = {'air11_1': (13.2, 268, 61.54, 4e-1, 30000),
            'air11_2': (19.2, 239, 19.2780, 1e-1, 100000),
            'air11_3': (26.2, 227, 9.9044, 1e-1, 500000),
            'air5_1': (13.2, 268, 61.54, 1e-1, 5000),
            'air5_2': (19.2, 239, 19.2780, 1e-1, 100000),
            'air5_3': (26.2, 227, 9.9044, 1e-1, 500000),
            'mars_1': (23.58, 172.05, 29.95, 7e-2, 20000),
            'mars_2': (13.2, 241.15, 638.83, 5e-2, 50000)}

# PRESHOCK CONDITIONS
M_inf = 23.58  # freestream mach number
T_inf = 241.15  # freestream temperature [K]
P_inf = 638.83 # freestream pressure [Pa]

# CREATE THE MIXTURE
opts = mpp.MixtureOptions("Mars_19")
opts.setStateModel("ChemNonEq1T")
mix = mpp.Mixture(opts)
mix.equilibrate(T_inf, P_inf)
rho_inf = mix.density()
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

def RH_jumpYi_Eq(state) :
    global u_inf,T_inf,P_inf,h_inf,rho_inf, mix

    u, T, P = state                    #current state variables

    err_value = 1e+20
    if P < 0 or T < 0 or u < 0:
        return [err_value, err_value, err_value]


    mix.equilibrate(T,P)
    h = mix.mixtureHMass()
    rho = mix.density()


    eq1 = rho*u - rho_inf*u_inf
    eq2 = P_inf+rho_inf*u_inf*u_inf-P-rho*u*u
    eq3 = h_inf + 0.5*u_inf*u_inf - h - 0.5*u*u
    return [eq1, eq2, eq3]

solution = fsolve(rk_jump, np.array([u_2, T_2, P_2]))
solution_eq = fsolve(RH_jumpYi_Eq,np.array([300, 5000, 440000]),maxfev=10000)
print('''
INITIAL SOLUTION 
U = {} [m/s]
T = {} [K]
P = {} [Pa] 

POST SHOCK
U = {} [m/s]
T = {} [K]
P = {} [Pa]

EQUILIBRIUM
U = {} [m/s]
T = {} [K]
P = {} [Pa]
NU = {:e}
LAMBDA = {:e}
'''.format(u_inf, T_inf, P_inf,*solution,*solution_eq,mix.averageHeavyCollisionFreq(),mix.meanFreePath()))
