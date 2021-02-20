import numpy as np
import matplotlib.pyplot as plt

# i_j = np.arange(-20,21,1)
# d = 5
# w_max = 27
# weights = w_max * np.exp(-np.abs(i_j)/d)
# weights[20] = 0
#
# plt.figure(figsize=(5,2.5))
# plt.plot(i_j, weights)
# plt.xticks(np.arange(-20, 21, 5))
# plt.xlim((-20,20))
# plt.ylim((0,25))
# plt.xlabel('$i-j$')
# plt.ylabel('$w_{ij}$')
# plt.savefig('weights_initial',bbox_inches='tight')
# plt.show()

d = 1
f = 0.6
U = 0.6
tau_stf = 200
tau_std = 500
r = 40
delta_t = 10

time_plot = [0]
f_plot = [f]
d_plot = [d]
fd_plot = [f * d]
for t_step in range(1000):
	d_next = ((1 - d) / tau_std - r*d*f) * delta_t
	f_next = ((U - f) / tau_stf + U * (1 - f) * r) * delta_t
	time_plot.append(t_step * delta_t)
	f_plot.append(f_next)
	d_plot.append(d_next)
	print(d_next)
	fd_plot.append((d_next * f_next))
	d = d_next
	f = f_next
	print(t_step / 10)

plt.plot(time_plot, f_plot)
plt.plot(time_plot, d_plot)
plt.plot(time_plot, fd_plot)
plt.show()