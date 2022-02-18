import numpy as np

with open('../../data/797_coarse.mshEXPERIMENTAL.npy', 'rb')as f:
    p =np.load(f)
    t = np.load(f)
    pg = np.load(f, allow_pickle=True).item()

for key in pg:
    print(key)
    print(pg[key]['nodes_and_normals'])
    print()