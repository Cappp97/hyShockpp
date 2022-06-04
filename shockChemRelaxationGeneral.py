# IMPORT LIBRARIES
import mutationpp as mpp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math as mt
import matplotlib.pyplot as plt


print('Succesfully imported shared libraries')

# PRESHOCK CONDITIONS
M_inf = 16 # freestream mach number
T_inf = 247.02  # freestream temperature [K]
P_inf = 21.96  # freestream pressure [Pa]

# MIXTURES
opts1T = mpp.MixtureOptions("air_11")
opts1T.setStateModel("ChemNonEq1T")
opts1T.setThermodynamicDatabase("RRHO")
mix1T = mpp.Mixture(opts1T)

optsTTv = mpp.MixtureOptions("air_11")
optsTTv.setStateModel("ChemNonEqTTv")
optsTTv.setThermodynamicDatabase("RRHO")
mixTTv = mpp.Mixture(optsTTv)


print('Successfully initialized required mixture')

# EQUILIBRIUM CONDITIONS 1T
mix1T.equilibrate(T_inf, P_inf)  # set the mixture in chemical equilibrium before the shock
rho_inf = mix1T.density()  # freestream density [kg/m^3]
print("The freestream density is {}".format(rho_inf))
c_inf = mix1T.equilibriumSoundSpeed()  # sound speed
u_inf = M_inf * mix1T.equilibriumSoundSpeed()  # air speed
gamma = mix1T.mixtureEquilibriumGamma()  # specific heat ratio
h_inf = mix1T.mixtureHMass()  # enthalpy
Yi = mix1T.Y()  # mass ratios
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
def RH_jump1T(state):
    global u_inf, rho_inf, h_inf, mix1T, Yi  # global variables

    u, T, P = state  # current state

    # This piece of code is necessary to avoid unphysical conditions therefore avoiding stability and convergence issues
    # If the solver ends up in an unphysical state the code immediately returns a very high value
    err_value = 1e+17
    if P < 0 or T < 0 or u < 0:
        return [err_value, err_value, err_value]

    # Set the current mixture state
    # because we are using frozen chemistry, the concentrations will be fixed
    # Note: Mixture is not in chemical equilibrium
    mix1T.setState(Yi, [P, T], 2)

    # get conserved variables
    rho = mix1T.density()
    h = mix1T.mixtureHMass()

    # Rankine Hugoniot jump conditions
    eq1 = u_inf * rho_inf - rho * u
    eq2 = P_inf + rho_inf * u_inf ** 2 - P - rho * u ** 2
    eq3 = h_inf + 0.5 * u_inf ** 2 - h - 0.5 * u ** 2
    return [eq1, eq2, eq3]

# Frozen chemistry solution
print('Solving the frozen chemistry shock model')
solution1T = fsolve(RH_jump1T, np.array([u_0, T_0, P_0]))
err = np.linalg.norm(RH_jump1T(solution1T), 2)  # L2 norm of the residual
init_to_print = np.round([u_inf, T_inf, P_inf], 2)
sol_to_print = np.round(solution1T, 2)
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


# EQUILIBRIUM CONDTIONS
mixTTv.equilibrate(T_inf, P_inf)  # set the mixture in chemical equilibrium before the shock
rho_inf = mixTTv.density()  # freestream density [kg/m^3]
print("The freestream density is {}".format(rho_inf))
c_inf = mixTTv.equilibriumSoundSpeed()  # sound speed
u_inf = M_inf * mixTTv.equilibriumSoundSpeed()  # air speed
gamma = mixTTv.mixtureEquilibriumGamma()  # specific heat ratio
h_inf = mixTTv.mixtureHMass()  # enthalpy
Yi = mixTTv.Y()  # mass ratios
ev_inf = mixTTv.mixtureEnergies()[1]
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
def RH_jumpTTv(state):
    global u_inf, rho_inf, h_inf, ev_inf, mixTTv, Yi  # global variables

    u, T, Tv, P = state  # current state

    # This piece of code is necessary to avoid unphysical conditions therefore avoiding stability and convergence issues
    # If the solver ends up in an unphysical state the code immediately returns a very high value
    err_value = 1e+17
    if P < 0 or T < 0 or u < 0 or Tv < 0:
        return [err_value, err_value, err_value, err_value]

    # Set the current mixture state
    # because we are using frozen chemistry, the concentrations will be fixed
    # Note: Mixture is not in chemical equilibrium
    mixTTv.setState(Yi, [P, T, Tv], 2)

    # get conserved variables
    rho = mixTTv.density()
    h = mixTTv.mixtureHMass()
    ev = mixTTv.mixtureEnergies()[1]

    # Rankine Hugoniot jump conditions
    eq1 = u_inf * rho_inf - rho * u
    eq2 = P_inf + rho_inf * u_inf ** 2 - P - rho * u ** 2
    eq3 = h_inf + 0.5 * u_inf ** 2 - h - 0.5 * u ** 2
    eq4 = rho_inf * u_inf * ev_inf - rho * u * ev
    return [eq1, eq2, eq3, eq4]



# Frozen chemistry solution
print('Solving the frozen chemistry shock model')
solutionTTv = fsolve(RH_jumpTTv, np.array([u_0, T_0, T_inf, P_0]))
err = np.linalg.norm(RH_jumpTTv(solutionTTv), 2)  # L2 norm of the residual
init_to_print = np.round([u_inf, T_inf, T_inf, P_inf], 2)
sol_to_print = np.round(solutionTTv, 2)
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
x = np.linspace(0,1e-1,50000)
dx = x[1]-x[0]

def RH_jumpYi_1T(state,*args) :
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


def RH_jumpYi_TTv(state,*args) :
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





# Solution 1T case
us1T = [solution1T[0]]
Ts1T = [solution1T[1]]
Ps1T = [solution1T[2]]
rhos1T = [mix1T.density()]
hs1T = [mix1T.mixtureHMass()]
Yis1T = [Yi]
mix1T = mpp.Mixture(opts1T)
mix1T.equilibrate(P_inf,T_inf)

for i in range(1,len(x)) :
    print("Section {0} out of {1} sum of densities {2}".format(i,len(x),np.sum(Yis1T[i-1])))
    #OLD MIXTURE
    mix1T.setState(Yis1T[i-1],[Ps1T[i-1],Ts1T[i-1]],2)

    #COMPUTE PRODUCTION RATES
    wi = mix1T.netProductionRates()
    args = ([us1T[i-1],Ts1T[i-1],Ps1T[i-1],hs1T[i-1],rhos1T[i-1],wi],
            Yis1T[i-1],
            mix1T) #argument list

    state = [us1T[i-1],Ts1T[i-1],Ps1T[i-1],*(Yis1T[i-1]+wi*dx/(rhos1T[i-1]*us1T[i-1]))] #initial guess
    sol1T = fsolve(RH_jumpYi_1T,state,args,xtol = 1e-12)
    us1T.append(sol1T[0])
    Ts1T.append(sol1T[1])
    Ps1T.append(sol1T[2])
    Yis1T.append(sol1T[3:])

    mix1T.setState(Yis1T[i],[Ps1T[i],Ts1T[i]],2)
    rhos1T.append(mix1T.density())
    hs1T.append(mix1T.mixtureHMass())



# Solution TTv case
usTTv = [solutionTTv[0]]
TsTTv = [solutionTTv[1]]
TvsTTv = [solutionTTv[2]]
PsTTv = [solutionTTv[3]]
rhosTTv = [mixTTv.density()]
hsTTv = [mixTTv.mixtureHMass()]
evsTTv = [ev_inf]
YisTTv = [Yi]
mixTTv = mpp.Mixture(optsTTv)
mixTTv.equilibrate(P_inf,T_inf)


for i in range(1,len(x)) :
    print("Section {0} out of {1} sum of densities {2}".format(i,len(x),np.sum(YisTTv[i-1])))
    #OLD MIXTURE
    mixTTv.setState(YisTTv[i-1],[PsTTv[i-1],TsTTv[i-1], TvsTTv[i-1]], 2)

    #COMPUTE PRODUCTION RATES
    wi = mixTTv.netProductionRates()
    OmegaV = mixTTv.energyTransferSource()
    args = ([usTTv[i-1],TsTTv[i-1], TvsTTv[i-1],PsTTv[i-1],hsTTv[i-1], evsTTv[i-1],rhosTTv[i-1],wi, OmegaV],
            YisTTv[i-1],
            mixTTv) #argument list

    state = [usTTv[i-1], TsTTv[i-1], TvsTTv[i-1], PsTTv[i-1], *(YisTTv[i-1]+wi*dx/(rhosTTv[i-1]*usTTv[i-1]))] #initial guess
    solTTv = fsolve(RH_jumpYi_TTv,state,args,xtol = 1e-12)
    usTTv.append(solTTv[0])
    TsTTv.append(solTTv[1])
    TvsTTv.append(solTTv[2])
    PsTTv.append(solTTv[3])
    YisTTv.append(solTTv[4:])

    mixTTv.setState(YisTTv[i],[PsTTv[i],TsTTv[i],TvsTTv[i]],2)
    rhosTTv.append(mixTTv.density())
    hsTTv.append(mixTTv.mixtureHMass())
    evsTTv.append(mixTTv.mixtureEnergies()[1])


# ===========================  Plots ==============================#


# --------------------------------------------------- Density -------------------------------------------------------- #
plt.figure(1)
plt.plot(x,rhos1T,'k-',lw=2)
plt.plot(x,rhosTTv,'g-',lw=2)
plt.title("Density")
plt.xlabel("x [m]")
plt.ylabel(r'$\rho [kg/m^3]$')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

# ------------------------------------------------ Temperatures ------------------------------------------------------ #
plt.figure(2)
plt.plot(x,Ts1T,'k-',lw=2)
plt.plot(x,TsTTv,'r-',lw=2)
plt.plot(x,TvsTTv, 'r--', lw=2)
plt.title("Temperature")
plt.xlabel("x [m]")
plt.ylabel(r'$T [K]$')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

# -------------------------------------------------- Velocity -------------------------------------------------------- #
plt.figure(3)
plt.plot(x,us1T,'k-',lw=2)
plt.plot(x,usTTv,'k-',lw=2)
plt.title("Velocity")
plt.xlabel("x [m]")
plt.ylabel(r'$u [m/s]$')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

# --------------------------------------------------- Pressure ------------------------------------------------------- #
plt.figure(4)
plt.plot(x,Ps1T,'k-',lw=2)
plt.plot(x,PsTTv,'r-',lw=2)
plt.title("Pressure")
plt.xlabel("x [m]")
plt.ylabel(r'$P [Pa]$')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

# --------------------------------------------------- Species -------------------------------------------------------- #
plt.figure(5)
plt.ylim((1e-6,4))
plt.semilogy(x,Yis1T, '-')
plt.semilogy(x, YisTTv, '--')
plt.grid()
plt.legend([mix1T.speciesName(i) for i in range(mix1T.nSpecies())])
plt.title("Species Mass Fractions")
plt.xlabel("x [m]")
plt.ylabel(r'$Y_i$ [-]')
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

plt.show()

