import numpy as np
from viz import plot_sol

fpath = '../out/797_coarse_sol.npy'

with open(fpath, 'rb') as f:
    p = np.load(f)
    t = np.load(f)
    u = np.load(f)

print(p.shape)
print(t.shape)
print(u.shape)
plot_sol(p, t, u)
