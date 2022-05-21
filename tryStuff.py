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
T0 = 300
T0v = 300

mix.equilibrate(T0, P0)
dens = mix.densities()
en = mix.mixtureEnergyMass()*mix.density()
mix.setState(dens, en, 5
             )

print(mix.Tv())

print(mix.energyTransferSource())

ht = mix.mixtureEnergyMass() + mix.P()/mix.density()
print(ht)


