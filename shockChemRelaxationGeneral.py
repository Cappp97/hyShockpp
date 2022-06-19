# ============================================= IMPORT USEFUL LIBRARIES ============================================== #
import mutationpp as mpp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math as mt
import matplotlib.pyplot as plt


print('Successfully imported shared libraries')
# ==================================================================================================================== #

# ======================================== CASE STORING IN A DICTIONARY ============================================== #
#                                                                                                                      #
#                               Case options: air_1, air_2, air_3, mars_1, mars_2                                      #
#                               CaseData = { key: (M_inf, T_inf, P_inf, DomainEnd, Nx)}                                #
#                                                                                                                      #

Case = 'air11_1'
NumberCase = "CASE2"

Folder  = {'air11_1': "AIR",
           'air11_2': "AIR",
           'air11_3': "AIR",
           'air5_1' : "AIR",
           'air5_2' : "AIR",
           'air5_3' : "AIR",
           'mars_1': "AIR",
           'mars_2': "AIR",
          }

Mixtures = {'air11_1': "air_11",
            'air11_2': "air_11",
            'air11_3': "air_11",
            'air5_1' : "air_5",
            'air5_2' : "air_5",
            'air5_3' : "air_5",
            'mars_1': "Mars_19",
            'mars_2': "Mars_19",
            }

CaseData = {'air11_1': (13.2, 268, 61.54, 4e-1, 30000),
            'air11_2': (19.2, 239, 19.2780, 1e-1, 100000),
            'air11_3': (26.2, 227, 9.9044, 1e-1, 500000),
            'air5_1': (13.2, 268, 61.54, 1e-1, 10000),
            'air5_2': (19.2, 239, 19.2780, 1e-1, 100000),
            'air5_3': (26.2, 227, 9.9044, 1e-1, 500000),
            'mars_1': (23.58, 172.05, 29.95, 1e-1, 50000),
            'mars_2': (13.2, 241.15, 638.83, 5e-4, 50000)}

filename1T = "DATA/{:}/{:}/{:}"
filenameTTv = "DATA/{:}/{:}/{:}"
# ==================================================================================================================== #


# ================================================= SETTING THE CASE ================================================= #
data = CaseData[Case]

# --------------------------------------------- Pre-Shock Conditions ------------------------------------------------- #
M_inf = data[0]                     # freestream mach number
T_inf = data[1]                     # freestream temperature [K]
P_inf = data[2]                     # freestream pressure [Pa]

# ----------------------------------------------------- Mesh --------------------------------------------------------- #
x = np.linspace(0, data[3], data[4])
dx = x[1]-x[0]

# ==================================================================================================================== #



# =============================================== SETTING THE MIXTURE ================================================ #

# ---------------------------------------------- One Temperature Model ----------------------------------------------- #
opts1T = mpp.MixtureOptions(Mixtures[Case])
opts1T.setStateModel("ChemNonEq1T")
opts1T.setThermodynamicDatabase("RRHO")
mix1T = mpp.Mixture(opts1T)

# ---------------------------------------------- Two Temperature Model ----------------------------------------------- #
optsTTv = mpp.MixtureOptions(Mixtures[Case])
optsTTv.setStateModel("ChemNonEqTTv")
optsTTv.setThermodynamicDatabase("RRHO")
mixTTv = mpp.Mixture(optsTTv)

print('Successfully initialized required mixture')

# ---------------------------------------------- One Temperature Model ----------------------------------------------- #
optsEq = mpp.MixtureOptions(Mixtures[Case])
optsEq.setStateModel("ChemNonEq1T")
optsEq.setThermodynamicDatabase("RRHO")
mixEq = mpp.Mixture(optsEq)

# ==================================================================================================================== #



# ===================================== COMPUTE THE POST-SHOCK CONDITIONS - 1T ======================================= #

# --------------------------------------------- Equilibrium Conditions ----------------------------------------------- #
mix1T.equilibrate(T_inf, P_inf)                     # set the mixture in chemical equilibrium before the shock
rho_inf = mix1T.density()                           # freestream density [kg/m^3]
print("The freestream density is {}".format(rho_inf))
c_inf = mix1T.equilibriumSoundSpeed()               # sound speed
u_inf = M_inf * mix1T.equilibriumSoundSpeed()       # air speed
gamma = mix1T.mixtureEquilibriumGamma()             # specific heat ratio
h_inf = mix1T.mixtureHMass()                        # enthalpy
Yi = mix1T.Y()                                      # mass ratios

print('Initial conditions computed')


# -------------------------------- First Guess from polytropic ideal std. air mixture -------------------------------- #
P_0 = ((2 * gamma / (gamma + 1)) * (M_inf ** 2 - 1) + 1) * P_inf
rho_0 = rho_inf * (1 - (2 / (gamma + 1)) * (1 - (1 / M_inf ** 2))) ** (-1)
u_0 = u_inf - (2 * c_inf / (gamma + 1)) * (M_inf - (1 / M_inf))
T_0 = P_0 / (rho_0 * 287)


# -----------------------------------  Non-Linear Solver for RH jump conditions -------------------------------------- #
# The state vector is composed by the variables u, T, P                                                                #
# (Note: any other set of variables can be used, these are not the conserved variables)                                #
# Returns the residual of the RH jump condition at a given state.                                                      #
# The nonlinear solver will try to set the residual to zero.                                                           #

def RH_jump1T(state):
    global u_inf, rho_inf, h_inf, mix1T, Yi     # global variables

    u, T, P = state                             # current state

    # This piece of code is necessary to avoid unphysical conditions, i.e. avoiding stability and convergence issues.
    # If the solver ends up in an unphysical state, the code immediately returns a very high value.

    err_value = 1e+17
    if P < 0 or T < 0 or u < 0:
        return [err_value, err_value, err_value]

    # Set the current mixture state.
    # because we are using frozen chemistry, the concentrations will be fixed.
    # Note: Mixture is not in chemical equilibrium.
    mix1T.setState(Yi, [P, T], 2)

    # Get conserved variables
    rho = mix1T.density()
    h = mix1T.mixtureHMass()

    # Rankine Hugoniot jump conditions
    eq1 = u_inf * rho_inf - rho * u
    eq2 = P_inf + rho_inf * u_inf ** 2 - P - rho * u ** 2
    eq3 = h_inf + 0.5 * u_inf ** 2 - h - 0.5 * u ** 2
    return [eq1, eq2, eq3]

# ------------------------------------ Frozen-chemistry Rankine-Hugoniot solution ------------------------------------ #
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
# ==================================================================================================================== #


# ===================================== COMPUTE THE POST-SHOCK CONDITIONS - 2T ======================================= #

# --------------------------------------------- Equilibrium Conditions ----------------------------------------------- #
mixTTv.equilibrate(T_inf, P_inf)                    # set the mixture in chemical equilibrium before the shock
rho_inf = mixTTv.density()                          # freestream density [kg/m^3]
print("The freestream density is {}".format(rho_inf))
c_inf = mixTTv.equilibriumSoundSpeed()              # sound speed
u_inf = M_inf * mixTTv.equilibriumSoundSpeed()      # air speed
gamma = mixTTv.mixtureEquilibriumGamma()            # specific heat ratio
h_inf = mixTTv.mixtureHMass()                       # enthalpy
Yi = mixTTv.Y()                                     # mass ratios
ev_inf = mixTTv.mixtureEnergies()[1]
print('Initial conditions computed')

# -------------------------------- First Guess from polytropic ideal std. air mixture -------------------------------- #
P_0 = ((2 * gamma / (gamma + 1)) * (M_inf ** 2 - 1) + 1) * P_inf
rho_0 = rho_inf * (1 - (2 / (gamma + 1)) * (1 - (1 / M_inf ** 2))) ** (-1)
u_0 = u_inf - (2 * c_inf / (gamma + 1)) * (M_inf - (1 / M_inf))
T_0 = P_0 / (rho_0 * 287)


# -----------------------------------  Non-Linear Solver for RH jump conditions -------------------------------------- #
# The state vector is composed by the variables u, T, Tv, P                                                            #
# (Note: any other set of variables can be used, these are not the conserved variables)                                #
# Returns the residual of the RH jump condition at a given state.                                                      #
# The nonlinear solver will try to set the residual to zero.                                                           #

def RH_jumpTTv(state):
    global u_inf, rho_inf, h_inf, ev_inf, mixTTv, Yi    # global variables

    u, T, Tv, P = state                                 # current state

    # This piece of code is necessary to avoid unphysical conditions, i.e. avoiding stability and convergence issues.
    # If the solver ends up in an unphysical state, the code immediately returns a very high value.
    err_value = 1e+17
    if P < 0 or T < 0 or u < 0 or Tv < 0:
        return [err_value, err_value, err_value, err_value]

    # Set the current mixture state.
    # because we are using frozen chemistry, the concentrations will be fixed.
    # Note: Mixture is not in chemical equilibrium.
    mixTTv.setState(Yi, [P, T, Tv], 2)

    # Get conserved variables
    rho = mixTTv.density()
    h = mixTTv.mixtureHMass()
    ev = mixTTv.mixtureEnergies()[1]

    # Rankine Hugoniot jump conditions
    eq1 = u_inf * rho_inf - rho * u
    eq2 = P_inf + rho_inf * u_inf ** 2 - P - rho * u ** 2
    eq3 = h_inf + 0.5 * u_inf ** 2 - h - 0.5 * u ** 2
    eq4 = rho_inf * u_inf * ev_inf - rho * u * ev
    return [eq1, eq2, eq3, eq4]


# ------------------------------------ Frozen-chemistry Rankine-Hugoniot solution ------------------------------------ #
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
# ==================================================================================================================== #



# ======================================= COMPUTE THE NEW STATE VECTOR - 1T ========================================== #
def RH_jumpYi_1T(state, *args) :
    global dx
    u0, T0, P0, h0, rho0, wi0 = args[0]     #old variables
    Yi0 = args[1]

    air = args[2]                           #mixture

    state = np.array(state)
    u, T, P = state[0:3]                    #current state variables
    Yi = state[3:]

    air.setState(Yi, [P, T], 2)
    wi = air.netProductionRates()
    h = air.mixtureHMass()
    rho = air.density()

    eq1 = rho*u - rho0*u0
    eq2 = P0+rho0*u0*u0-P-rho*u*u
    eq3 = h0 + 0.5*u0*u0 - h - 0.5*u*u
    eqs = rho0*Yi0*u0 - rho*Yi*u + 0.5*dx*(wi0+wi)
    return [eq1, eq2, eq3, *eqs]


# ======================================= COMPUTE THE NEW STATE VECTOR - 2T ========================================== #
def RH_jumpYi_TTv(state,*args) :
    global dx
    u0, T0, T0v, P0, h0, ev0, rho0, wi0, OmegaV0 = args[0]  #old variables
    Yi0 = args[1]

    air = args[2] #mixture

    state = np.array(state)
    u, T, Tv, P = state[0:4]                                   #current state variables
    Yi = state[4:]

    air.setState(Yi, [P, T, Tv], 2)
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

    return [eq1, eq2, eq3, eq4, *eqs]

# Equilibrium
def RH_jumpYi_Eq(state,*args) :
    global u_inf,T_inf,P_inf,h_inf,rho_inf

    air = args[0]                           #mixture

    u, T, P = state                    #current state variables

    err_value = 1e+17
    if P < 0 or T < 0 or u < 0:
        return [err_value, err_value, err_value]


    air.equilibrate(T,P)
    h = air.mixtureHMass()
    rho = air.density()


    eq1 = rho*u - rho_inf*u_inf
    eq2 = P_inf+rho_inf*u_inf*u_inf-P-rho*u*u
    eq3 = h_inf + 0.5*u_inf*u_inf - h - 0.5*u*u
    return [eq1, eq2, eq3]
# ==================================================================================================================== #


# ================================================= Solution 1T case ================================================= #
us1T = [solution1T[0]]
Ts1T = [solution1T[1]]
Ps1T = [solution1T[2]]
rhos1T = [mix1T.density()]
hs1T = [mix1T.mixtureHMass()]
Yis1T = [Yi]
mix1T = mpp.Mixture(opts1T)
mix1T.equilibrate(P_inf, T_inf)

for i in range(1,len(x)) :
    print("Section {0} out of {1} sum of densities {2}".format(i, len(x), np.sum(Yis1T[i-1])))
    # ------------------------------------------------ Old Mixture --------------------------------------------------- #
    mix1T.setState(Yis1T[i-1], [Ps1T[i-1], Ts1T[i-1]], 2)

    # --------------------------------------- Compute the production rates ------------------------------------------- #
    wi = mix1T.netProductionRates()

    # ----------------------------------------- Set new arguments list ----------------------------------------------- #
    args = ([us1T[i-1], Ts1T[i-1], Ps1T[i-1], hs1T[i-1], rhos1T[i-1], wi],
            Yis1T[i-1],
            mix1T)

    # --------------------------------------------- Set initial guess ------------------------------------------------ #
    state = [us1T[i-1], Ts1T[i-1], Ps1T[i-1], *(Yis1T[i-1]+wi*dx/(rhos1T[i-1]*us1T[i-1]))]

    # -------------------------------------------- Solve for new state ----------------------------------------------- #
    sol1T = fsolve(RH_jumpYi_1T, state, args, xtol=1e-12)

    # -------------------------------------------- Store new solution ------------------------------------------------ #
    us1T.append(sol1T[0])
    Ts1T.append(sol1T[1])
    Ps1T.append(sol1T[2])
    Yis1T.append(sol1T[3:])

    mix1T.setState(Yis1T[i], [Ps1T[i], Ts1T[i]], 2)
    rhos1T.append(mix1T.density())
    hs1T.append(mix1T.mixtureHMass())


Yis1T = np.array(Yis1T)
rhos1T = np.array(rhos1T)
Ps1T = np.array(Ps1T)
us1T = np.array(us1T)
Ts1T = np.array(Ts1T)

"""Yis1T_tosave = np.array2string(Yis1T, precision=64, separator=',')
rhos1T_tosave = np.array2string(rhos1T, precision=64, separator=',')
us1T_tosave = np.array2string(us1T, precision=64, separator=',')
Ts1T_tosave = np.array2string(Ts1T, precision=64, separator=',')
Ps1T_tosave = np.array2string(Ps1T, precision=64, separator=',')

# Save temperature 1T
f = open(filename1T.format(Folder[Case], NumberCase, "Temperature1T.txt"), "w")
f.write(Ts1T_tosave)
f.write("\n")
f.close()
# Save total density 1T
f = open(filename1T.format(Folder[Case], NumberCase, "Density1T.txt"), "w")
f.write(rhos1T_tosave)
f.write("\n")
f.close()
# Save pressure 1T
f = open(filename1T.format(Folder[Case], NumberCase, "Pressure1T.txt"), "w")
f.write(Ps1T_tosave)
f.write("\n")
f.close()
# Save Velocity
f = open(filename1T.format(Folder[Case], NumberCase, "Velocity1T.txt"), "w")
f.write(us1T_tosave)
f.write("\n")
f.close()
# Save Species
f = open(filename1T.format(Folder[Case], NumberCase, "Species1T.txt"), "w")
f.write(Yis1T_tosave)
f.write("\n")
f.close()"""



# ================================================= Solution 2T case ================================================= #
usTTv = [solutionTTv[0]]
TsTTv = [solutionTTv[1]]
TvsTTv = [solutionTTv[2]]
PsTTv = [solutionTTv[3]]
rhosTTv = [mixTTv.density()]
hsTTv = [mixTTv.mixtureHMass()]
evsTTv = [ev_inf]
YisTTv = [Yi]
mixTTv = mpp.Mixture(optsTTv)
mixTTv.equilibrate(P_inf, T_inf)


for i in range(1,len(x)) :
    print("Section {0} out of {1} sum of densities {2}".format(i,len(x),np.sum(YisTTv[i-1])))
    # ------------------------------------------------ Old Mixture --------------------------------------------------- #
    mixTTv.setState(YisTTv[i-1], [PsTTv[i-1], TsTTv[i-1], TvsTTv[i-1]], 2)

    # --------------------------------------- Compute the production rates ------------------------------------------- #
    wi = mixTTv.netProductionRates()
    OmegaV = mixTTv.energyTransferSource()

    # ----------------------------------------- Set new arguments list ----------------------------------------------- #
    args = ([usTTv[i-1], TsTTv[i-1], TvsTTv[i-1], PsTTv[i-1], hsTTv[i-1], evsTTv[i-1], rhosTTv[i-1], wi, OmegaV],
            YisTTv[i-1],
            mixTTv)

    # --------------------------------------------- Set initial guess ------------------------------------------------ #
    state = [usTTv[i-1], TsTTv[i-1], TvsTTv[i-1], PsTTv[i-1], *(YisTTv[i-1]+wi*dx/(rhosTTv[i-1]*usTTv[i-1]))]

    # -------------------------------------------- Solve for new state ----------------------------------------------- #
    solTTv = fsolve(RH_jumpYi_TTv, state, args, xtol=1e-12)

    # -------------------------------------------- Store new solution ------------------------------------------------ #
    usTTv.append(solTTv[0])
    TsTTv.append(solTTv[1])
    TvsTTv.append(solTTv[2])
    PsTTv.append(solTTv[3])
    YisTTv.append(solTTv[4:])

    mixTTv.setState(YisTTv[i], [PsTTv[i], TsTTv[i], TvsTTv[i]], 2)
    rhosTTv.append(mixTTv.density())
    hsTTv.append(mixTTv.mixtureHMass())
    evsTTv.append(mixTTv.mixtureEnergies()[1])

YisTTv = np.array(YisTTv)
rhosTTv = np.array(rhosTTv)
PsTTv = np.array(PsTTv)
usTTv = np.array(usTTv)
TsTTv = np.array(TsTTv)

"""YisTTv_tosave = np.array2string(YisTTv, precision=64, separator=',')
rhosTTv_tosave = np.array2string(rhosTTv, precision=64, separator=',')
usTTv_tosave = np.array2string(usTTv, precision=64, separator=',')
TsTTv_tosave = np.array2string(TsTTv, precision=64, separator=',')
PsTTv_tosave = np.array2string(PsTTv, precision=64, separator=',')

# Save temperature 1T
f = open(filenameTTv.format(Folder[Case], NumberCase, "TemperatureTTv.txt"), "w")
f.write(Ts1T_tosave)
f.write("\n")
f.close()
# Save total density 1T
f = open(filenameTTv.format(Folder[Case], NumberCase, "DensityTTv.txt"), "w")
f.write(rhos1T_tosave)
f.write("\n")
f.close()
# Save pressure 1T
f = open(filenameTTv.format(Folder[Case], NumberCase, "PressureTTv.txt"), "w")
f.write(Ps1T_tosave)
f.write("\n")
f.close()
# Save Velocity
f = open(filenameTTv.format(Folder[Case], NumberCase, "VelocityTTv.txt"), "w")
f.write(us1T_tosave)
f.write("\n")
f.close()
# Save Species
f = open(filenameTTv.format(Folder[Case], NumberCase, "SpeciesTTv.txt"), "w")
f.write(Yis1T_tosave)
f.write("\n")
f.close()"""

# ==================================================================================================================== #
x0 = np.array([us1T[-1],Ts1T[-1],Ps1T[-1]])
solEq = fsolve(RH_jumpYi_Eq,x0,mixEq)
uEq,Teq,Peq = solEq
mixEq.equilibrate(Teq, Peq)
rhoEq = mixEq.density()




# ====================================================== Plots ======================================================= #
#
# ---------------------------------------------------- PLOT LEGEND --------------------------------------------------- #
#
#                                           - 1 Temperature Model plots:
#                                                   1. Linestyle:   '--'
#                                                   2. Color:      red i.e. 'r'
#
#                                           - 2 Temperatures Model plots:
#                                                  1. Linestyle:   '-'
#                                                  2. Color        blue i.e. 'b'
#
# -------------------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------- Species Colors ----------------------------------------------------- #
SpeciesColors = {'O2': (255, 0, 0), 'N2': (0, 255, 0), 'NO': (0, 0, 255), 'O': (238, 130, 238),
                 'N': (255, 128, 0), 'N+': (128, 0, 0), 'O+': (0, 238, 238), 'NO+': (240, 255, 255),
                 'N2+': (205, 205, 0), 'O2+': (85, 26, 139), 'e-': (0, 0, 0), 'CO2': (25, 25, 112),
                 'CO': (51, 161, 201), 'CN': (105, 139, 34), 'C2': (199, 21, 133), 'C': (139, 69, 0),
                 'CO+': (61, 145, 64), 'CN+': (233, 150, 122), 'C+': (255, 193, 37)}


# --------------------------------------------------- Density -------------------------------------------------------- #
plt.figure(1)
plt.plot(x, rhos1T, 'r--', lw=2, label='1T')
plt.plot(x, rhosTTv, 'b-', lw=2, label='TTv')
plt.plot([x[0],x[-1]],[rhoEq,rhoEq],'g--',lw=2,label='Thermochemical Equilibrium')
plt.title("Density", fontsize=18)
plt.xlabel("x [m]", fontsize=18)
plt.ylabel(r'$\rho   [kg/m^3]$', fontsize=18)
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.legend()

# ------------------------------------------------ Temperatures ------------------------------------------------------ #
plt.figure(2)
plt.plot(x, Ts1T, 'r--', lw=2, label='One T model')
plt.plot(x, TsTTv, 'b-', lw=2, label='Two T model - Trt')
plt.plot(x, TvsTTv, 'c-', lw=2, label='Two T model - Tv')
plt.plot([x[0],x[-1]],[Teq,Teq],'g--',lw=2,label='Thermochemical Equilibrium')
plt.title("Temperature", fontsize=18)
plt.xlabel("x [m]", fontsize=18)
plt.ylabel(r'$T [K]$', fontsize=18)
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.legend()

# -------------------------------------------------- Velocity -------------------------------------------------------- #
plt.figure(3)
plt.plot(x, us1T, 'r--', lw=2, label='1T')
plt.plot(x, usTTv, 'b-', lw=2, label='TTv')
plt.plot([x[0],x[-1]],[uEq,uEq],'g--',lw=2,label='Thermochemical Equilibrium')
plt.title("Velocity", fontsize=18)
plt.xlabel("x [m]", fontsize=18)
plt.ylabel(r'$u [m/s]$', fontsize=18)
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.legend()

# --------------------------------------------------- Pressure ------------------------------------------------------- #
plt.figure(4)
plt.plot(x, Ps1T, 'r--', lw=2, label='1T')
plt.plot(x, PsTTv, 'b-', lw=2, label='TTv')
plt.plot([x[0],x[-1]],[Peq,Peq],'g--',lw=2,label='Thermochemical Equilibrium')
plt.title("Pressure", fontsize=18)
plt.xlabel("x [m]", fontsize=18)
plt.ylabel(r'$P [Pa]$', fontsize=18)
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.legend()

# --------------------------------------------------- Species -------------------------------------------------------- #
plt.figure(5)
plt.ylim((1e-6, 4))
for i in range(mix1T.nSpecies()):

    species = mix1T.speciesName(i)
    RGB = SpeciesColors[species]
    RGB = tuple(map(lambda t: round(t/255., 2), RGB))

    plt.semilogy(x, Yis1T[:, i], '-', color=RGB, alpha=1.0,label=species)
    plt.semilogy(x, YisTTv[:, i], '--', color=RGB, alpha=1.0)

plt.grid()
plt.title("Species Mass Fractions", fontsize=18)
plt.xlabel("x [m]", fontsize=18)
plt.ylabel(r'$Y_i$ [-]', fontsize=18)
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='--')

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.legend(loc='best')

plt.show()

# ==================================================================================================================== #

