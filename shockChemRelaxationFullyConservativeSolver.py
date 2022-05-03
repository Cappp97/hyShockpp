# IMPORT LIBRARIES
import mutationpp as mpp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math as mt

print('Succesfully imported shared libraries')

# PRESHOCK CONDITIONS
M_inf = 16 # freestream mach number
T_inf = 247.02  # freestream temperature [K]
P_inf = 21.96  # freestream pressure [Pa]

# MIXTURE
opts = mpp.MixtureOptions("air_11_omar")
opts.setStateModel("ChemNonEq1T")
opts.setThermodynamicDatabase("RRHO")
mix = mpp.Mixture(opts)

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

# MESH
x = np.linspace(0,5E-3,10000)
dx = x[1]-x[0]

def rk_jumpYi(state,*args) :
    global dx
    u0,T0,P0,h0,rho0,wi0 = args[0] #old variables
    Yi0 = args[1]

    air = args[2] #mixture

    state = np.array(state)
    u,T,P = state[0:3] #current state variables
    Yi = state[3:]

    air.setState(Yi,[P,T],2)
    wi = air.netProductionRates()
    h = air.mixtureHMass()
    rho = air.density()

    eq1 = rho*u - rho0*u0
    eq2 = P0+rho0*u0*u0-P-rho*u*u
    eq3 = h0 + 0.5*u0*u0 - h - 0.5*u*u
    eqs = rho0*Yi0*u0 - rho*Yi*u + 0.5*dx*(wi0+wi)
    return [eq1,eq2,eq3,*eqs]


us = [solution[0]]
Ts = [solution[1]]
Ps = [solution[2]]
rhos = [mix.density()]
hs = [mix.mixtureHMass()]
Yis = [Yi]
mix = mpp.Mixture(opts)
mix.equilibrate(P_inf,T_inf)
for i in range(1,len(x)) :
    print("Section {0} out of {1} sum of densities {2}".format(i,len(x),np.sum(Yis[i-1])))
    #OLD MIXTURE
    mix.setState(Yis[i-1],[Ps[i-1],Ts[i-1]],2)

    #COMPUTE PRODUCTION RATES
    wi = mix.netProductionRates()
    args = ([us[i-1],Ts[i-1],Ps[i-1],hs[i-1],rhos[i-1],wi],
            Yis[i-1],
            mix) #argument list

    state = [us[i-1],Ts[i-1],Ps[i-1],*(Yis[i-1]+wi*dx/(rhos[i-1]*us[i-1]))] #initial guess
    sol = fsolve(rk_jumpYi,state,args,xtol = 1e-12)
    us.append(sol[0])
    Ts.append(sol[1])
    Ps.append(sol[2])
    Yis.append(sol[3:])

    mix.setState(Yis[i],[Ps[i],Ts[i]],2)
    rhos.append(mix.density())
    hs.append(mix.mixtureHMass())


import matplotlib.pyplot as plt


plt.ylim((1e-6,4))
plt.semilogy(x,Yis)
plt.grid()
plt.legend([mix.speciesName(i) for i in range(mix.nSpecies())])
plt.title("Species Mass Fractions")
plt.xlabel("x [m]")
plt.ylabel(r'$Y_i$ [-]')
plt.show()

plt.plot(x,rhos,'k-',lw=2)
plt.title("Density")
plt.xlabel("x [m]")
plt.ylabel(r'$\rho [kg/m^3]$')
plt.show()

plt.plot(x,Ts,'k-',lw=2)
plt.title("Temperature")
plt.xlabel("x [m]")
plt.ylabel(r'$T [K]$')
plt.show()

plt.plot(x,Ps,'k-',lw=2)
plt.title("Pressure")
plt.xlabel("x [m]")
plt.ylabel(r'$P [Pa]$')
plt.show()

plt.plot(x,us,'k-',lw=2)
plt.title("Velocity")
plt.xlabel("x [m]")
plt.ylabel(r'$u [m/s]$')
plt.show()







