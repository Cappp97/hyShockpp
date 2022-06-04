import mutationpp as mpp
import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math as mt
import matplotlib.pyplot as plt

print('Succesfully imported shared libraries')

# PRESHOCK CONDITIONS
M_inf = 16# freestream mach number
T_inf = 247.02  # freestream temperature [K]
P_inf = 21.96  # freestream pressure [Pa]



# MIXTURE
opts = mpp.MixtureOptions("air_11")
opts.setStateModel("ChemNonEqTTv")
opts.setThermodynamicDatabase("RRHO")
mix = mpp.Mixture(opts)

print('Succesfully initialized required mixture')

# EQUILIBRIUM CONDTIONS
mix.equilibrate(T_inf, P_inf)  # set the mixture in chemical equilibrium before the shock
rho_inf = mix.density()  # freestream density [kg/m^3]
print("The freestream density is {}".format(rho_inf))
c_inf = mix.equilibriumSoundSpeed()  # sound speed
u_inf = M_inf * mix.equilibriumSoundSpeed()  # air speed
gamma = mix.mixtureEquilibriumGamma()  # specific heat ratio
h_inf = mix.mixtureHMass()  # enthalpy
Yi = mix.Y()  # mass ratios
ev_inf = mix.mixtureEnergies()[1]
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
    global u_inf, rho_inf, h_inf, ev_inf, mix, Yi  # global variables

    u, T, Tv, P = state  # current state

    # This piece of code is necessary to avoid unphysical conditions therefore avoiding stability and convergence issues
    # If the solver ends up in an unphyscal state the code immediatly returns a very high value
    err_value = 1e+17
    if P < 0 or T < 0 or u < 0 or Tv < 0:
        return [err_value, err_value, err_value, err_value]

    # Set the current mixture state
    # because we are using frozen chemistry, the concentrations will be fixed
    # Note: Mixture is not in chemical equilibrium
    mix.setState(Yi, [P, T, Tv], 2)

    # get conserved variables
    rho = mix.density()
    h = mix.mixtureHMass()
    ev = mix.mixtureEnergies()[1]

    # Rankine Hugoniot jump conditions
    eq1 = u_inf * rho_inf - rho * u
    eq2 = P_inf + rho_inf * u_inf ** 2 - P - rho * u ** 2
    eq3 = h_inf + 0.5 * u_inf ** 2 - h - 0.5 * u ** 2
    eq4 = rho_inf * u_inf * ev_inf - rho * u * ev
    return [eq1, eq2, eq3, eq4]



# Frozen chemistry solution
print('Solving the frozen chemistry shock model')
solution = fsolve(rk_jump, np.array([u_0, T_0, T_inf, P_0]))
err = np.linalg.norm(rk_jump(solution), 2)  # L2 norm of the residual
init_to_print = np.round([u_inf, T_inf, T_inf, P_inf], 2)
sol_to_print = np.round(solution, 2)
const_to_print = np.round([u_0, T_0, T_inf, P_0], 2)

print('''
NONLINEAR SHOCK SOLVER SOLUTION
INITIAL CONDITION            SOLUTION               CONSTANT CP SOLUTION
U = {0} [m/s]               U = {4} [m/s]           U = {8} [m/S]            
T = {1} [K]                 T = {5} [K]             T = {9} [K]
Tv = {2} [K]                 Tv = {6} [K]             Tv = {10} [K]
P = {3} [Pa]                P = {7} [Pa]            P = {11} [Pa]
ERROR
eps = {12}
'''.format(*init_to_print, *sol_to_print, *const_to_print, err))


# MESH
Nx = 50000
x = np.linspace(0,1e-1,Nx)
dx = x[1]-x[0]

print(dx)

def rk_jumpYi(state,*args) :
    global dx
    u0, T0, T0v, P0, h0, ev0, rho0, wi0, OmegaV0 = args[0] #old variables
    Yi0 = args[1]

    air = args[2] #mixture

    state = np.array(state)
    u,T,Tv,P = state[0:4] #current state variables
    Yi = state[4:]

    air.setState(Yi,[P,T,Tv],2)
    wi = air.netProductionRates()
    OmegaV = air.energyTransferSource()
    h = air.mixtureHMass()
    rho = air.density()
    ev = air.mixtureEnergies()[1]

    eq1 = rho*u - rho0*u0
    eq2 = P0+rho0*u0*u0-P-rho*u*u
    eq3 = h0 + 0.5*u0*u0 - h - 0.5*u*u
    eq4 = rho0 * u0 * ev0 - rho * u * ev + 0.5*dx*(OmegaV0 + OmegaV)
    eqs = rho0*Yi0*u0 - rho*Yi*u + 0.5*dx*(wi0+wi)

    return [eq1,eq2,eq3,eq4,*eqs]


us = [solution[0]]
Ts = [solution[1]]
Tvs = [solution[2]]
Ps = [solution[3]]
rhos = [mix.density()]
hs = [mix.mixtureHMass()]
evs = [ev_inf]
Yis = [Yi]
mix = mpp.Mixture(opts)
mix.equilibrate(P_inf,T_inf)


for i in range(1,len(x)) :
    print("Section {0} out of {1} sum of densities {2}".format(i,len(x),np.sum(Yis[i-1])))
    #OLD MIXTURE
    mix.setState(Yis[i-1],[Ps[i-1],Ts[i-1], Tvs[i-1]], 2)

    #COMPUTE PRODUCTION RATES
    wi = mix.netProductionRates()
    OmegaV = mix.energyTransferSource()
    args = ([us[i-1],Ts[i-1], Tvs[i-1],Ps[i-1],hs[i-1], evs[i-1],rhos[i-1],wi, OmegaV],
            Yis[i-1],
            mix) #argument list

    state = [us[i-1], Ts[i-1], Tvs[i-1], Ps[i-1], *(Yis[i-1]+wi*dx/(rhos[i-1]*us[i-1]))] #initial guess
    sol = fsolve(rk_jumpYi,state,args,xtol = 1e-12)
    us.append(sol[0])
    Ts.append(sol[1])
    Tvs.append(sol[2])
    Ps.append(sol[3])
    Yis.append(sol[4:])

    mix.setState(Yis[i],[Ps[i],Ts[i],Tvs[i]],2)
    rhos.append(mix.density())
    hs.append(mix.mixtureHMass())
    evs.append(mix.mixtureEnergies()[1])



plt.figure(1)
plt.ylim((1e-6,4))
plt.semilogy(x,Yis)
plt.grid()
plt.legend([mix.speciesName(i) for i in range(mix.nSpecies())])
plt.title("Species Mass Fractions")
plt.xlabel("x [m]")
plt.ylabel(r'$Y_i$ [-]')


plt.figure(2)
plt.plot(x,rhos,'k-',lw=2)
plt.title("Density")
plt.xlabel("x [m]")
plt.ylabel(r'$\rho [kg/m^3]$')


plt.figure(3)
plt.plot(x,Ts,'k-',lw=2)
plt.plot(x,Tvs, 'r--', lw=2)
plt.title("Temperature")
plt.xlabel("x [m]")
plt.ylabel(r'$T [K]$')


plt.figure(4)
plt.plot(x,Ps,'k-',lw=2)
plt.title("Pressure")
plt.xlabel("x [m]")
plt.ylabel(r'$P [Pa]$')


plt.figure(5)
plt.plot(x,us,'k-',lw=2)
plt.title("Velocity")
plt.xlabel("x [m]")
plt.ylabel(r'$u [m/s]$')

plt.show()
