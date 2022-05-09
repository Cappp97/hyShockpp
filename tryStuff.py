# IMPORT LIBRARIES
import numpy as np


f = np.array([1,2,3])
f = np.hstack([f,f[0]])

flux_diff = -f[0:-1] + f[1:]
print(flux_diff)

