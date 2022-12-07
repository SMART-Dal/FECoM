import matplotlib.pyplot as plt

def normalised(lst):
    min_x = min(lst)
    max_x = max(lst)

    return [(x - min_x) / (max_x - min_x) for x in lst]

data_lines = []
with open('perf.txt', 'r') as f:
    data_lines = f.read().splitlines(True)

# use the last n data points
n = 100

cpu_energies = normalised([float(line.strip(' ').split(';')[1]) for line in data_lines[2::2][-n:]])
ram_energies = normalised([float(line.strip(' ').split(';')[1]) for line in data_lines[3::2][-n:]])

time = [i/2 for i in range(len(cpu_energies))]

fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(nrows=2, ncols=2)

ax1.set_title("CPU Energy over time")
ax1.plot(time, cpu_energies, 'o')

ax2.set_title("RAM Energy over time")
ax2.plot(time, ram_energies, 'o')

ax3.set_title("RAM Energy vs CPU Energy")
ax3.plot(cpu_energies, ram_energies, 'o')

ax4.set_title("RAM+CPU Energy over time")
ram_plus_cpu = [sum(x) for x in zip(cpu_energies, ram_energies)]
ax4.plot(time, ram_plus_cpu, 'o')


plt.show()