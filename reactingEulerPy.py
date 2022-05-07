# Import some useful libraries

from scipy.integrate import quad
from scipy.integrate import solve_ivp
import numpy as np
import math as mt
import mutationpp as mpp

# ============================================ Definition of the mixture ============================================= #
opts    = mpp.MixtureOptions("air_5")
opts.setStateModel("ChemNonEq1T")
opts.setThermodynamicDatabase("RRHO")
mix     = mpp.Mixture(opts)


# ============================================== Definition of the Mesh ============================================== #
#                                                                                                                      #
#                                                                                                                      #
#      |---------o--------|---------o---------|---------o---------|---------o---------|---------o---------|            #
#                                          x_(i-0.5)   x_i     x_(i+0.5)                                               #
#                                                                                                                      #
# ==================================================================================================================== #

# Spatial Discretization
Length          = 0.1                                                       # Length of the domain from 0.0
Nx              = 1000                                                      # Number of space discretizations
x               = np.linspace(0.0, Length, Nx)                              # Cell Boundaries of the domain
h               = Length/(Nx-1)                                             # Cell dimension

meshPoints  = 0.5*(x[0:-1] + x[1:])                                         # Control Points of the cells

# Time Discretization
dt              = 1e-7                                                      # Time step
Nt              = 1000                                                      # Number of time iterations

t = (0., dt*Nt)


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
            index   = cellIdx * (2 + nEnergyEqs + nSpecies) + 2
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


Patm            = 1.01325e5
P0              = lambda xx: 2*Patm if (xx > 0.03 or xx < 0.06) else Patm                  # Pressure
T0              = lambda xx: 300.                                                           # Temperature
u0              = 0.                                                                        # Velocity
solution        = np.zeros((2 + mix.nEnergyEqns() + mix.nSpecies()) * len(meshPoints))
Pressure0       = np.zeros((Nt, len(meshPoints)))


for i in range(len(x)-1):

    # Retrieve the equilibrium conditions within the cell
    T0_cell             = quad(T0, x[i], x[i+1])[0]/h
    P0_cell             = quad(P0, x[i], x[i+1])[0]/h
    mix.equilibrate(T0_cell, P0_cell)

    rho                 = lambda xx: mix.density()
    momentum            = lambda xx: mix.density()*u0
    totalEnergy         = mix.mixtureEnergyMass()
    rhoEt               = lambda xx: totalEnergy*mix.density() + 0.5*u0**2
    rhos                = mix.densities()

    Pressure0[0, i]     = quad(P0, x[i], x[i+1])[0]/h

    # Retrieve indices for the solution vector
    Rho_idx             = DOF_HANDLER(i, 0, mix.nSpecies(), mix.nEnergyEqns(), 'Density')
    Momentum_idx        = DOF_HANDLER(i, 0, mix.nSpecies(), mix.nEnergyEqns(), 'Momentum')
    Energy_idx          = DOF_HANDLER(i, 0, mix.nSpecies(), mix.nEnergyEqns(), 'Energy')


    solution[Rho_idx]       = quad(rho, x[i], x[i + 1])[0] / h
    solution[Momentum_idx]  = quad(momentum, x[i], x[i + 1])[0] / h
    solution[Energy_idx]    = quad(rhoEt, x[i], x[i + 1])[0] / h
    
    for j in range(mix.nSpecies()):
        Rhos_idx            = DOF_HANDLER(i, j, mix.nSpecies(), mix.nEnergyEqns(), 'Specie')
        dens                = lambda xx: rhos[j]
        solution[Rhos_idx]  = quad(dens, x[i], x[i + 1])[0] / h

del rho, rhos, momentum, rhoEt, dens, Rhos_idx, Rho_idx, Energy_idx, Momentum_idx, totalEnergy, i, j

# =============================================== Time integration =================================================== #

nVar               = 2 + mix.nEnergyEqns() + mix.nSpecies()
def dwdt(t, solution):

    dw              = np.zeros_like(solution)
    fluxes_prev     = np.zeros_like(solution)
    ws_integrated   = np.zeros_like(solution)

    for i in range(len(meshPoints)):

        # Actual cell variables
        rho         = solution[nVar * i]
        rhou        = solution[nVar * i + 1]
        rhoet       = solution[nVar * i + 2]
        rhos        = solution[nVar * i + 2 + mix.nEnergyEqns(): nVar * i + 2 + mix.nEnergyEqns() + mix.nSpecies()]
        if (mix.nEnergyEqns() == 2):
            rhoetv  = solution[nVar * i + 3]

        # Previous cell variables
        iPrev = len(meshPoints) - 1 if i == 0 else i - 1

        rho_prev    = solution[nVar * iPrev]
        rhou_prev   = solution[nVar * iPrev + 1]
        rhoet_prev  = solution[nVar * iPrev + 2]
        rhos_prev   = solution[nVar * iPrev + 2 + mix.nEnergyEqns(): nVar * iPrev + 2 + mix.nEnergyEqns() + mix.nSpecies()]
        if (mix.nEnergyEqns() == 2):
            rhoetv_prev  = solution[nVar * iPrev + 3]


        # ---------------------------------- Setting the state inside the cell  -------------------------------------- #
        mix.setState(rhos, rhoet - 0.5 * rhou*rhou/rho, 0)
        P           = mix.P()
        T           = mix.T()

        # ------------------------------------ Assembly of the Roe's Matrix ------------------------------------------ #
        A           = np.zeros((nVar, nVar))

        # Definition of the Roe's averaged state
        rhocap      = mt.sqrt(rho*rho_prev)
        ucap        = ((mt.sqrt(rho)*rhou/rho) + (mt.sqrt(rho_prev)*rhou_prev/rho_prev))/(mt.sqrt(rho) + mt.sqrt(rho_prev))
        etotcap     = ((mt.sqrt(rho)*rhoet/rho) + (mt.sqrt(rho_prev)*rhoet_prev/rho_prev))/(mt.sqrt(rho) + mt.sqrt(rho_prev))
        rhoscap     = ((mt.sqrt(rho)*rhos/rho) + (mt.sqrt(rho_prev)*rhos_prev/rho_prev))/(mt.sqrt(rho) + mt.sqrt(rho_prev))

        # Thermodynamic derivatives
        Ys          = mix.Y()
        R           = 8.31446261815324                                              # Universal Gas Constant
        Ri          = [R / mix.speciesMw(k) for k in range(mix.nSpecies())]         # Species Gas Constants
        dP_dRho     = sum([Ys[i] * Ri[i] * T for k in range(mix.nSpecies())])       # Derivative of pressure wrt density
        dP_dT       = rho * sum([Ys[i] * Ri[i] for k in range(mix.nSpecies())])     # Derivative of pressure wrt energy

        # Compute terms inside Roe's matrix -- Notation: fij = df_i/dw_j, i & j = 0, .. , nVar-1
        # First row
        f00         = 0.                                #   NOTE:   The terms related to the species
        f01         = 1.                                #           are computed separately later
        f02         = 0.                                #           inside a for loop
        f03         = 0.   # This is only for TTv model #

        # Second row
        f10         = -(rhocap*ucap)**2/rhocap**2 + dP_dRho
        f11         = 2*rhocap*ucap/rhocap
        f12         = 0.
        f13         = 0.    # This is only for TTv model

        # Third row
        f20         = -(rhocap*ucap*rhocap*etotcap)/rhocap**2 + (rhocap*ucap/rhocap)*(dP_dRho - P/rhocap**2)
        f21         = rhocap*etotcap/rhocap + P/rhocap
        f22         = 0.
        f23         = 0.    # This is only for TTv model

        # Fourth row  --    This is only for TTv model
        # TO BE DONE!!

        # Species rows
        f0123j = np.zeros(mix.nSpecies())  # This is the end of the 0,1,2 and 3 rows
        fij = []
        for j in range(mix.nSpecies()):
            fj0     = -rhoscap[j]*rhocap*ucap/rhocap**2
            fj1     = rhoscap[j] / rhocap
            fj2     = 0.
            fj3     = 0.                                # This is only for TTv model

            fjs     = np.zeros((mix.nSpecies()))
            fjs[j]  = rhocap*ucap/rhocap

            if (mix.nEnergyEqns() == 1):
                rowj    = np.concatenate((np.array([fj0, fj1, fj2]), fjs))
            else:
                rowj    = np.concatenate((np.array([fj0, fj1, fj2, fj3]), fjs))

            fij     = np.concatenate((fij,rowj))

        fij = np.reshape( fij, (mix.nSpecies,nVar))

        if (mix.nEnergyEqns() == 1):
            A0      = np.hstack([f00, f01, f02], f0123j)
            A1      = np.hstack([f10, f11, f12], f0123j)
            A2      = np.hstack([f20, f21, f22], f0123j)
            A       = np.block([[A0], [A1], [A2], [fij]])
        else:
            A0      = np.hstack([f00, f01, f02, f03], f0123j)
            A1      = np.hstack([f10, f11, f12, f13], f0123j)
            A2      = np.hstack([f20, f21, f22, f23], f0123j)
            A3      = np.zeros((1,nVar))
            A       = np.block([[A0], [A1], [A2], [A3], [fij]])


        a = np.linalg.det(A)

        # ---------------------------------------- Computation of the fluxes ----------------------------------------- #
        if (mix.nEnergyEqns() == 1):

            # Left fluxes
            fL0     = rhou_prev
            fL1     = rhou_prev*rhou_prev/rho_prev + P
            fL2     = rhou_prev*rhoet_prev/rho_prev + P*rhou_prev/rho_prev
            fL3     = rhos_prev*rhou_prev/rho_prev

            fL      = np.concatenate((fL0, fL1, fL2, fL3))

            # Right fluxes
            fR0     = rhou
            fR1     = rhou * rhou / rho + P
            fR2     = rhou * rhoet / rho + P * rhou / rho
            fR3     = rhos * rhou / rho

            fR      = np.concatenate((fR0, fR1, fR2, fR3))

            # Total Flux at previous interface
            variable_prev   = solution[nVar*iPrev:nVar*(iPrev+1)]
            variable        = solution[nVar*i:nVar*(i+1)]
            fluxes_prev[nVar * i: nVar*(i+1)] = 0.5*(fL + fR) - 0.5*a*(variable - variable_prev)
        else:
            # TO BE DONE
            continue


        # --------------------------------- Computation of the net production rates ---------------------------------- #
        ws          = mix.netProductionRates()
        ws_integrated[nVar * i + 2 + mix.nEnergyEqns():nVar * i + 2 + mix.nEnergyEqns() + mix.nSpecies()] = ws


    # ------------------------------- Computation of the RHS of the semi-discrete formulation ------------------------ #
    fluxes = -fluxes_prev[0:-1] + fluxes_prev[1:]

    dw = ws_integrated - fluxes/h
    return dw

print('Integrating the system')
sol = solve_ivp(dwdt, t, solution, method='RK23', t_eval=np.linspace(t[0], t[1], Nt))

import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0, :])
plt.show()
print(mix.T(), mix.P())





















