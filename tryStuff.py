# IMPORT LIBRARIES
import mutationpp as mpp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math as mt

# MIXTURE
opts = mpp.MixtureOptions("air_5")
opts.setStateModel("ChemNonEqTTv")
opts.setThermodynamicDatabase("RRHO")
mix = mpp.Mixture(opts)

P0 = 101325
T0r = 300
T0v = 300

mix.equilibrate(T0r,P0)
print(mix.mixtureEnergyMass())
