import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def N_GCE (E): 
  constant = 8.6*10**14
  parentheses = E
  exponent = 0.27-(0.27*np.log10(E))
  return constant*(parentheses**exponent)

lower_N_GCE_limit = 1
upper_N_GCE_limit = 100

result_N_GCE, error_N_GCE = quad(N_GCE, lower_N_GCE_limit,upper_N_GCE_limit)

# Load data
djdvr = np.loadtxt('E_C_Cusp_Total_Data.txt', delimiter='\t', skiprows=1)
radial_data = djdvr[:, 0]          # radius values
annihilation_data = djdvr[:, 1]    # dJ/dV(r) values

# Interpolator: dJ/dV as function of radius
djdv_of_r = interp1d(radial_data, annihilation_data, kind='linear', fill_value='extrapolate')

# Geometry
d = 803.0
theta_range = np.linspace(0, 0.244346095, 80)  # radians (0–14 deg)
r_max = radial_data.max()

# Integrand: function of LOS distance ℓ, with theta as a parameter
def integrand(l, theta):
    r0 = d * np.sin(theta)
    r_val = np.sqrt(l*l + r0*r0)   # radius at this LOS point
    return float(djdv_of_r(r_val))

# Loop over theta
results = np.zeros_like(theta_range)
errors  = np.zeros_like(theta_range)

for i, theta in enumerate(theta_range):
    r0 = d * np.sin(theta)
    if r0 >= r_max:
        # Outside the tabulated region
        results[i] = 0.0
        errors[i]  = 0.0
        continue
    l_max = np.sqrt(r_max**2 - r0**2)   # maximum LOS distance inside r_max
    results[i], errors[i] = quad(integrand, 0.0, l_max, args=(theta,))

print(results)

plt.yscale('log')
plt.xscale('log')
plt.plot(np.linspace(0,14,80),results*2*(1/4*np.pi)*result_N_GCE)
plt.show()
