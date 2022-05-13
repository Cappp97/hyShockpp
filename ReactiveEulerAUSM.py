# Import useful libraries
import numpy as np
import mutationpp as mpp
import math as mt
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ============================================ Definition of the mixture ============================================= #
opts = mpp.MixtureOptions("air_5")
opts.setStateModel("ChemNonEq1T")
opts.setThermodynamicDatabase("RRHO")
mix = mpp.Mixture(opts)

# ============================================== Definition of the Mesh ============================================== #
#                                                                                                                      #
#                                                                                                                      #
#      |---------o--------|---------o---------|---------o---------|---------o---------|---------o---------|            #
#                                          x_(i-0.5)   x_i     x_(i+0.5)                                               #
#                                                                                                                      #
# ==================================================================================================================== #

# Spatial Discretization
Length = 0.1  # Length of the domain from 0.0
Nx = 1000  # Number of space discretizations
x = np.linspace(0.0, Length, Nx)  # Cell Boundaries of the domain
dx = Length / (Nx - 1)  # Cell dimension

meshPoints = 0.5 * (x[0:-1] + x[1:])  # Control Points of the cells

# Time Discretization
dt = 1e-7  # Time step
Nt = 2  # Number of time iterations

tEnd = dt * Nt


# ======================================= Definition of the initial conditions ======================================= #
#                                                                                                                      #
# The solution array is composed by:                                                                                   #
#                                                                                                                      #
#   sol0(0)             = density                                                                                      #
#   sol0(1)             = density*velocity                                                                             #
#   sol0(2)             = density*totalEnergy                                                                          #
#   sol0(2 + nSpecies:) = density_s                                                                                    #
#                                                                                                                      #
#   NOTE: For the access to the variable, stored in the solution array, follow this dof_handler:                       #
#                                                                                                                      #
#                          Density idx :                                                                               #
#              cell:   i    ------->   index = i * (2 + mix.nEnergyEqs() + nSpecies)                                   #
#                          Momentum idx :                                                                              #
#              cell:   i    ------->   index = i * (2 + mix.nEnergyEqs() + nSpecies) + 1                               #
#                          Energy idx :                                                                                #
#              cell:   i    ------->   index = i * (2 + mix.nEnergyEqs() + nSpecies) + 2                               #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
#                   NOTE: if "ChemNonEqTTv" is used                                                                    #
#                       Energy idx :                                                                                   #
#              cell:   i    ------->   index = i * (2 + mix.nEnergyEqs() + nSpecies) + 3                               #
# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
#                         Species idx :                                                                                #
#  (cell, species): (i,j)  ------->   index = 2 + mix.nEnergyEqs() + i * (2 + mix.nEnergyEqs() + mix.nSpecies() ) + j  #
#                                                                                                                      #
#                                                                                                                      #
# ==================================================================================================================== #

def DOF_HANDLER(cellIdx, specieIdx, nSpecies, nEnergyEqs, var):
    if (nEnergyEqs == 1):
        if (var == 'Density'):
            index = cellIdx * (2 + nEnergyEqs + nSpecies)

        elif (var == 'Momentum'):
            index = cellIdx * (2 + nEnergyEqs + nSpecies) + 1
        elif (var == 'Specie'):
            index = 2 + nEnergyEqs + cellIdx * (2 + nEnergyEqs + nSpecies) + specieIdx
        elif (var == 'Energy'):
            index = cellIdx * (2 + nEnergyEqs + nSpecies) + 2
    else:
        if (var == 'Density'):
            index = cellIdx * (2 + nEnergyEqs + nSpecies)
        elif (var == 'Momentum'):
            index = cellIdx * (2 + nEnergyEqs + nSpecies) + 1
        elif (var == 'Specie'):
            index = 2 + nEnergyEqs + cellIdx * (2 + nEnergyEqs + nSpecies) + specieIdx
        if (var == 'Energy Tr'):
            index = cellIdx * (2 + nEnergyEqs + nSpecies) + 2
        elif (var == 'Energy Tv'):
            index = cellIdx * (2 + nEnergyEqs + nSpecies) + 3

    return index


Patm = 1.01325e5
R = 8.31446261815324                                              # Universal Gas Constant
P0 = lambda xx: 2 * Patm if (xx > 0.03 and xx < 0.06) else Patm             # Pressure
T0 = lambda xx: 300.  # Temperature
u0 = 0.  # Velocity
solution = np.zeros((Nt, (2 + mix.nEnergyEqns() + mix.nSpecies()) * len(meshPoints)))
Pressure = np.zeros((Nt, len(meshPoints)))

for i in range(len(x) - 1):

    # Retrieve the equilibrium conditions within the cell
    T0_cell = quad(T0, x[i], x[i + 1])[0] / dx
    P0_cell = quad(P0, x[i], x[i + 1])[0] / dx
    mix.equilibrate(T0_cell, P0_cell)

    rho = lambda xx: mix.density()
    momentum = lambda xx: mix.density() * u0
    totalEnergy = mix.mixtureEnergyMass()
    rhoEt = lambda xx: totalEnergy * mix.density() + 0.5 * u0 ** 2
    Ys   = mix.Y()

    Pressure[0, i] = quad(P0, x[i], x[i + 1])[0] / dx

    # Retrieve indices for the solution vector
    Rho_idx = DOF_HANDLER(i, 0, mix.nSpecies(), mix.nEnergyEqns(), 'Density')
    Momentum_idx = DOF_HANDLER(i, 0, mix.nSpecies(), mix.nEnergyEqns(), 'Momentum')
    Energy_idx = DOF_HANDLER(i, 0, mix.nSpecies(), mix.nEnergyEqns(), 'Energy')

    solution[0, Rho_idx] = quad(rho, x[i], x[i + 1])[0] / dx
    solution[0, Momentum_idx] = quad(momentum, x[i], x[i + 1])[0] / dx
    solution[0, Energy_idx] = quad(rhoEt, x[i], x[i + 1])[0] / dx

    for j in range(mix.nSpecies()):
        Rhos_idx = DOF_HANDLER(i, j, mix.nSpecies(), mix.nEnergyEqns(), 'Specie')
        dens = lambda xx: solution[0, i*(2+mix.nEnergyEqns()+mix.nSpecies())]*Ys[j]
        solution[0, Rhos_idx] = quad(dens, x[i], x[i + 1])[0] / dx


del rho, Ys, momentum, rhoEt, dens, Rhos_idx, Rho_idx, Energy_idx, Momentum_idx, totalEnergy, i, j



# ============================================== AUSM Flux Discretization ============================================ #
def AUSM(q_prev, q):

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                   This is the indexing legend:                                                   #
    #                                                                                                                  #
    #             q_prev  = solution in (i-1) cell   ----> it's decomposed in (.)_L                                    #
    #             q       = solution in (i) cell     ----> it's decomposed in (.)_R                                    #
    #                                                                                                                  #
    #                                                                                                                  #
    #                   q composition:  [rho, momentum, total energy, rho*Ys1, ... , rho*Ysn]^(T)                      #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Compute Primary Variables

    # (i-1) cell
    rhoL   = q_prev[0]
    uL     = q_prev[1]/rhoL
    etL    = q_prev[2]/rhoL

    # (i) cell
    rhoR   = q[0]
    uR     = q[1] / rhoR
    etR    = q[2] / rhoR

    YsL    = np.zeros((mix.nSpecies()))
    YsR    = np.zeros((mix.nSpecies()))

    for j in range(mix.nSpecies()):
        YsL[j] = q_prev[2+mix.nEnergyEqns()+j]/rhoL
        YsR[j] = q[2+mix.nEnergyEqns()+j]/rhoR

    # Setting the state inside each cell

    # (i-1) cell
    mix.setState(rhoL*YsL, rhoL*etL - 0.5*rhoL*uL**2, 0)
    #mix.setState(rhoL*YsL, rhoL*etL, 0)
    htotL = mix.mixtureHMass()*rhoL
    pL = mix.P()
    TL = mix.T()
    cL = np.sqrt(mix.mixtureFrozenGamma()*R*TL)
    PsiL = np.hstack([[1, uL, htotL], YsL])


    # (i) cell
    mix.setState(rhoR * YsR, rhoR * etR - 0.5 * rhoR * uR ** 2, 0)
    #mix.setState(rhoR * YsR, rhoR * etR, 0)
    htotR = mix.mixtureHMass()*rhoR
    pR = mix.P()
    TR = mix.T()
    cR = np.sqrt(mix.mixtureFrozenGamma()*R*TR)
    PsiR = np.hstack([[1, uR, htotR], YsR])


    # ---------------------------------------------- AUSM ------------------------------------------------------------ #
    # Max speed of sound required for AUSM
    Cm = max(cL, cR)

    # Kind of sub/supersonic guard condition in computing contribution of different cells
    guardL = abs(uL)/Cm
    guardR = abs(uR)/Cm

    print("GuradL is: {:2f}\nGuardR is: {:2f}\n".format(guardL, guardR))

    # Pressure contribution
    if guardL < 1. :
        pL_plus = pL*(uL/Cm + 1)**2 * (2 - uL/Cm)/4
    else:
        pL_plus = pL*((uL + abs(uL))/(2*uL))

    if guardR < 1. :
        pR_minus = pR*(uR/Cm - 1)**2 * (2 + uR/Cm)/4
    else:
        pR_minus = pR*((uR + abs(uR))/(2*uR))

    p_Half = pL_plus + pR_minus

    p_Half_vec = np.hstack([[0, p_Half, 0], np.zeros(mix.nSpecies())])

    # Velocity contribution

    alphaR = (2*pR/rhoR)/(pL/rhoL + pR/rhoR)
    alphaL = (2*pL/rhoL)/(pL/rhoL + pR/rhoR)

    if guardL < 1. :
        uL_plus = alphaL*((uL + Cm)**2/(4*Cm) - (uL + abs(uL))/2.) + (uL + abs(uL))/2.
    else:
        uL_plus = (uL + abs(uL))/2.

    if guardR < 1. :
        uR_minus = alphaR*(-(uR - Cm)**2/(4*Cm) - (uR - abs(uR))/2.) + (uR - abs(uR))/2.
    else:
        uR_minus = (uR - abs(uR))/2.

    rhou_Half = uL_plus*rhoL + uR_minus*rhoR


    F_Half = 0.5*(rhou_Half*(PsiL + PsiR) - abs(rhou_Half)*(PsiR - PsiL)) + p_Half_vec
    return F_Half


# =============================================== Time integration =================================================== #


"""
def RungeKutta4(mix, rhoi, T, dt):

    mix.setState(rhoi, T, 0)
    wdot1 = mix.netProductionRates()

    rhoi += 0.5*np.multiply(dt, wdot1)
    mix.setState(rhoi, T, 0)
    wdot2 = mix.netProductionRates()

    rhoi += 0.5*np.multiply(dt,wdot2)
    mix.setState(rhoi,T,0)
    wdot3 = mix.netProductionRates() 
    
    rhoi += np.multiply(dt,wdot3)
    mix.setState(rhoi,T,0)
    wdot4 = mix.netProductionRates() 

    return 1./6. * np.multiply(dt, (np.array(wdot1) + 2 * np.array(wdot2) + 2 * np.array(wdot3) + np.array(wdot4)))

"""
nVar = 2 + mix.nEnergyEqns() + mix.nSpecies()
t = 0
dFprev = np.zeros_like(solution)
dFnext = np.zeros_like(solution)
for it in range(Nt-1):

    for i in range(len(meshPoints)):

        # Setting periodic boundary conditions
        iPrev = len(meshPoints)-1 if i == 0 else i - 1
        iNext = 0 if i == len(meshPoints)-1 else i + 1

        # Taking previous time-step solution
        qPrev = solution[it, iPrev*nVar:(iPrev+1)*nVar]
        q = solution[it, i * nVar:(i + 1) * nVar]
        qNext = solution[it, iNext * nVar:(iNext + 1) * nVar]

        # Computing the fluxes
        dFprev[it, i*nVar:(i+1)*nVar] = AUSM(qPrev, q)
        dFnext[it, i * nVar:(i + 1) * nVar] = AUSM(q, qNext)

        dF = dFnext - dFprev

    #print(dFprev)

    solution[it+1, :] = solution[it, :] + dF[it, :]*dt/dx


YinTime = np.zeros((Nt, mix.nSpecies()))
for sp in range(mix.nSpecies()):
    YinTime[:, sp] = solution[:, 2+mix.nEnergyEqns()+sp]
    #print((solution[:, 2+mix.nEnergyEqns()+sp::nVar]).shape)


times = np.linspace(0., Nt*dt, Nt)
plt.figure(5)
plt.plot(times, YinTime)
plt.show()

plotidx = -1

plt.figure(1)
plt.plot(meshPoints, solution[plotidx, 0::nVar])
plt.title("Density")

plt.figure(2)
plt.plot(meshPoints, solution[plotidx, 1::nVar])
plt.title("Momentum")

plt.figure(3)
plt.plot(meshPoints, solution[plotidx, 2::nVar])
plt.title("Energy")

plt.figure(4)
plt.plot(meshPoints, solution[plotidx, 2+mix.nEnergyEqns()+3::nVar])
plt.title("Xi")

plt.show()













