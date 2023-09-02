
# IEVDHM_V03 (Dr. Alejandro Zamora)
# Improved Eigen Value Decomposition Hankel Matrix - Version 02
# 15 abr 2021

import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot 	as plt
from scipy.linalg 			import hankel


def Energy_Threshold(s):
	sum_s 	= np.sum(s)
	sum_st 	= 0.0
	for modes, i in enumerate(s,  start = 1):
		sum_st 		= sum_st + i
		pc_sum_st 	= (sum_st / sum_s) * 100.0
		if pc_sum_st > 90.0: break

	return modes



def Mean_Sum(r, H):

	vec = []
	for i in range(r*2-1):
		temp = []
		for j in range(r):
			if (i >= j) and ((i - j) < r):
				temp.append(H[i - j, j])
		vec.append(sum(temp) / len(temp))

	return np.array(vec)


np.set_printoptions(suppress = True, precision = 3)

dt = 1.0/60.0
t = np.arange(0 , 30 + dt , dt)
N = len(t)

frq1 = 0.2
frq2 = 0.4
frq3 = 0.8

damp_rat1 = 0.8
damp_rat2 = 3.0
damp_rat3 = 0.3

damp1 = 2.0 * np.pi * frq1 * (damp_rat1 / 100.0)
damp2 = 2.0 * np.pi * frq2 * (damp_rat2 / 100.0)
damp3 = 2.0 * np.pi * frq3 * (damp_rat3 / 100.0)

s1 = (10.0) * np.exp(-damp1 * t) * np.cos((2 * np.pi * frq1 * t) + (np.pi / 4))
s2 = (15.0) * np.exp(-damp2 * t) * np.cos((2 * np.pi * frq2 * t) - (np.pi / 4))
s3 = (20.0) * np.exp(-damp3 * t) * np.cos((2 * np.pi * frq3 * t) + (np.pi / 3))
st = s1 + s2 + s3
st = st - np.mean(st)

# HANKEL MATRIX
r = int(np.around((N / 2.0), 0))

H = hankel(st)[0:r , 0:r]
U, s, v = np.linalg.svd(H)
S = np.diag(s)
V = v.T

modes = Energy_Threshold(s)

x = np.zeros([modes, N-2])
for m in range(0 , modes , 2):

	Hx = (S[m , m] * np.array([V[: , m]]).T @ np.array([V[: , m]])
			-  S[m+1 , m+1] * np.array([V[: , m+1]]).T  @ np.array([V[: , m+1]]))

	x[m,:] = Mean_Sum(r, Hx)
	
	Hxx = hankel(x[m,:])[0:r , 0:r]

	Ux, sx, vx = np.linalg.svd(Hxx)
	Sx = np.diag(sx)
	Vx = vx.T
	modesx = Energy_Threshold(sx)

	while (modesx / 2.0) > 1:
				
		i = 0
		Hx = (Sx[0 , 0] * np.array([Vx[: , 0]]).T @ np.array([Vx[: , 0]])
			-  Sx[1 , 1] * np.array([Vx[: , 1]]).T  @ np.array([Vx[: , 1]]))

		x[m,:] = Mean_Sum(r, Hx)

		Hxx = hankel(x[m,:])[0:r , 0:r]

		Ux, sx, vx = np.linalg.svd(Hxx)
		Sx = np.diag(sx)
		Vx = vx.T

		modesx = Energy_Threshold(sx)


# FIGURE
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = True)

ax1.plot(-x[2,:], dashes = [6,2], label = 's1')
ax1.plot(s1, label = 's1', color = 'black')
ax1.set_ylabel('s1')

ax2.plot(x[4,:], dashes = [6,2], label = 's2')
ax2.plot(s2, label = 's2', color = 'black')
ax2.set_ylabel('s2')

ax3.plot(-x[0,:], label = 's3')
ax3.plot(s3, label = 's3', color = 'black')
ax3.set_ylabel('s3')

xt = - x[0,:] - x[2,:] + x[4,:]
ax4.plot(xt, dashes = [6,2], label = 'st')
ax4.plot(st, label = 'st', color = 'black')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('st')

plt.show()


