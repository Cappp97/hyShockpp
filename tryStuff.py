# IMPORT LIBRARIES
import mutationpp as mpp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math as mt


a = np.genfromtxt("DATA/AIR/CASE2/Temperature1T.txt")

print(a.shape)
