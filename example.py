import mutationpp as mpp

import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines
from cycler import cycler


# Definition of the RK4 method for the time integration of the continuity eqs.
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


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

# Prepare the plot options
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
cc = cycler('linestyle', ['-', '--', ':', '-.', '-']) * cycler('color', ['r', 'g', 'b', 'm', 'k'])
plt.rc('axes', prop_cycle=cc)

plt.rc('text', usetex=True)
plt.rc('font', family='Helvetica')


# ======================== Choice of the mixture ============================= #
opts = mpp.MixtureOptions("air_11")
opts.setThermodynamicDatabase("NASA-9")
opts.setStateModel("ChemNonEq1T")
print("Mixture species:", opts.getSpeciesDescriptor())

mix = mpp.Mixture(opts)
ns = mix.nSpecies()

# ======================= Thermodynamic Initial Conditions ==================== #
T = 300
P = 10000

mix.equilibrate(T,P)
rhoi_eq = mix.densities()

Tinit = 15000
mix.setState(rhoi_eq,Tinit,1)
total_energy = mix.mixtureEnergyMass()*mix.density()


time = 0.
time_final = 1e-7
dt = 1.e-9
rhoi = rhoi_eq

y0 = np.divide(rhoi_eq,mix.density())
x0 = mix.convert_y_to_x(y0)
temperature = np.array(mix.T())
mass_fractions = np.array([y0])
mole_fraction =  np.array([x0])
time_reaction = np.array(time)

while (time < time_final):
    dt = min(dt * 1.0002, 1.e-10)
    drhoi = RungeKutta4(mix, rhoi, total_energy, dt)
    rhoi += drhoi
    time += dt
    mix.setState(rhoi,total_energy,0)
    y = mix.Y()
    x = mix.convert_y_to_x(np.array(y))
    mass_fractions = np.vstack((mass_fractions,[y]))
    mole_fraction =  np.vstack((mole_fraction,[x]))
    temperature = np.vstack((temperature,mix.T()))
    time_reaction  = np.vstack((time_reaction,time))

plt.figure()
for i in range(ns):
    plt.plot(time_reaction,mole_fraction[:,i],label=mix.speciesName(i))
ax = plt.gca()
ax.yaxis.set_label_coords(0.0, 1.00)
plt.ylabel('mole fraction', rotation=0)
plt.xlabel('time, s')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)  
labelLines(plt.gca().get_lines(), align=True, fontsize=10, ha='left')

plt.figure()
plt.plot(time_reaction,temperature)
ax = plt.gca()
ax.yaxis.set_label_coords(0.0, 1.00)
plt.ylabel('Temperature, K', rotation=0)
plt.xlabel('time, s')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)  
labelLines(plt.gca().get_lines(), align=True, fontsize=10, ha='left')
plt.show()


