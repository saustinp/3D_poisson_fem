import numpy as np
import matplotlib.pyplot as plt
import shelve
import re
import pandas as pd

h_array = np.zeros(8)
size_re = re.compile('CTETRA\': (\d+)')

with open('mesh_sizes.txt', 'r') as file:
    for count, line in enumerate(file.readlines()):
        match = re.search(size_re, line)
        h_array[count] = int(match.group(1))

h_array = (6*np.sqrt(2)*1/h_array)**(1/3)

convert_df = pd.read_csv('mesh_convert_time.txt')
sol_df = pd.read_csv('solution_benchmarking.txt')

wc_convert = convert_df['wc_time']
cpu_convert = convert_df['cpu_time']

wc_build_matrix = sol_df['wc_build_time']
cpu_build_matrix = sol_df['cpu_build_time']
wc_sol_backslash = sol_df['wc_solve_backslash_time']
cpu_sol_backslash = sol_df['cpu_solve_backslash_time']
wc_sol_cg = sol_df['wc_solve_cg_time']
cpu_sol_cg = sol_df['cpu_solve_cg_time']

def plot(h_array, y, title, filename):
    fig, ax = plt.subplots()
    ax.loglog(h_array, y)
    ax.invert_xaxis()
    plt.xlabel('h')
    plt.ylabel('seconds')
    plt.title(title)
    plt.savefig('benchmark_plots/'+filename+'.png')


# plot(h_array, wc_convert, 'Mesh convert time - wall clock', 'wc_convert')
# plot(h_array, cpu_convert, 'Mesh convert time - cpu clock', 'cpu_convert')
# plot(h_array[:-1], wc_build_matrix, 'Matrix build time - wall clock', "mat_build_wc")
# plot(h_array[:-1], cpu_build_matrix,
#      'Matrix build time - cpu', "mat_build_cpu")
# plot(h_array[:-1], wc_sol_backslash,
#      'Solution time using backslash - wall clock', "wc_sol_backslash")
# plot(h_array[:-1], cpu_sol_backslash,
#      'Solution time using backslash - cpu', "cpu_sol_backslash")
# plot(h_array[:-1], wc_sol_cg,
#      'Solution time using conjugate gradient - wall clock', "wc_sol_cg")
# plot(h_array[:-1], cpu_sol_cg,
#      'Solution time using conjugate gradient - cpu', "cpu_sol_cg")

fig, ax = plt.subplots()
ax.invert_xaxis()

ax.loglog(h_array[:-1], wc_sol_backslash, label='backslash')
ax.loglog(h_array[:-1], wc_sol_cg, label='conjugate gradient')
plt.xlabel('h')
plt.ylabel('seconds')
plt.title("Solution time compared using backslash and conjugate gradient")
plt.savefig('benchmark_plots/combined.png')
plt.legend()
# plt.show()