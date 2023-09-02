

# Improved Eigen Value Decomposition Hankel Matrix - Version 02
# 15 abr 2021

import sys, os
import time
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import hankel


def Energy_Threshold(s, energy_threshold):
	sum_s 	= np.sum(s)
	sum_st 	= 0.0
	for modes, i in enumerate(s,  start = 1):
		sum_st 		= sum_st + i
		pc_sum_st 	= (sum_st / sum_s) * 100.0
		if pc_sum_st > energy_threshold: break
	return modes


def Mean_Sum(r, H):
	vec = []
	for i in range(r+r-1):
		temp = []
		for j in range(r):
			if (i >= j) and ((i - j) < r):
				temp.append(H[i - j, j])
		vec.append(sum(temp) / len(temp))
	return np.array(vec)


np.set_printoptions(suppress = True, precision = 3)


# # # SYNTETHIC SIGNAL-----------------------------------------------------------------------------
# dt = 1.0/60.0
# t = np.arange(0 , 7.7 + dt , dt)

# frq1 = 0.26
# frq2 = 0.43
# frq3 = 0.64

# damp_rat1 = 5.0
# damp_rat2 = 0.3
# damp_rat3 = 10.0

# damp1 = 2.0 * np.pi * frq1 * (damp_rat1 / 100.0)
# damp2 = 2.0 * np.pi * frq2 * (damp_rat2 / 100.0)
# damp3 = 2.0 * np.pi * frq3 * (damp_rat3 / 100.0)

# s1 = (12.0) * np.exp(-damp1 * t) * np.cos((2 * np.pi * frq1 * t) + (np.pi / 4))
# s2 = (25.0) * np.exp(-damp2 * t) * np.cos((2 * np.pi * frq2 * t) - (np.pi / 3))
# s3 = (20.0) * np.exp(-damp3 * t) * np.cos((2 * np.pi * frq3 * t) + (2 * np.pi / 3))
# st = s1 + s2 + s3 + 20.0
# st = st - np.mean(st)


# # WECC SIGNAL-----------------------------------------------------------------------------

# csvData = csv.reader(open('Datos.csv'))
# Data = []
# for column in csvData:
# 	Data.append(column)

# to, yo = [], []
# for index, iData in enumerate(Data[1:]):
# 	to.append(float(iData[0]))
# 	yo.append([])

# 	for jData in iData[1:]:
# 		yo[index].append(float(jData))

# y_vec = np.array(yo)
# t_vec = np.array(to)
# dt = t_vec[1] - t_vec[0]

# ta = float(7.53)
# tb = float(ta + 8.4)
# pa, pb = int(round(ta / dt, 0)), int(round(tb / dt, 0) + 1)

# t 	= t_vec[pa : pb]
# st 	= y_vec[pa : pb , 0]
# st 	= st - np.mean(st)


# NEW ENGLAND ------------------------------------------------------------------------------

# csvData = csv.reader(open('new_england_16g.csv'))
# Data = []
# for column in csvData:
# 	Data.append(column)

# to, yo = [], []
# capt = True
# index = 0
# for iData in Data[1:]:

# 	# capt = not(capt)
# 	if capt == True:
# 		to.append(float(iData[0]))
# 		yo.append([])

# 		for jData in iData[1:]:
# 			yo[index].append(float(jData))
# 		index = index + 1



# y_vec = np.array(yo)
# t_vec = np.array(to)
# dt = t_vec[1] - t_vec[0]

# ta = float(2.0)
# tb = float(ta + 5.46)
# pa, pb = int(round(ta / dt, 0)), int(round(tb / dt, 0) + 1)

# t 	= t_vec[pa : pb]
# st 	= y_vec[pa : pb , 15]
# st 	= st - np.mean(st)


# MEX FREQ 28DIC2020 -------------------------------------------------------------------------

csvData = csv.reader(open('mex_freq_28dic2020.csv'))
Data = []
for column in csvData:
	Data.append(column)

to, yo = [], []
capt = True
index = 0
for iData in Data[1:]:

	# capt = not(capt)
	if capt == True:
		to.append(float(iData[0]))
		yo.append([])

		for jData in iData[1:]:
			yo[index].append(float(jData))
		index = index + 1



y_vec = np.array(yo)
t_vec = np.array(to)
dt = t_vec[1] - t_vec[0]

ta = float(120.0)
tb = float(ta + 20.0)
pa, pb = int(round(ta / dt, 0)), int(round(tb / dt, 0) + 1)

t 	= t_vec[pa : pb]
st 	= y_vec[pa : pb , :]
st 	= st - np.mean(st)


# HANKEL MATRIX //////////////////////////////////////////////////////////////////////////////
t0_ievdhm = time.time()

N = len(t)
r = int(np.around((N / 2.0), 0))

H = hankel(st)[0:r , 0:r]
U, s, v = np.linalg.svd(H)
S = np.diag(s)
V = v.T

modes = Energy_Threshold(s, 60.0)
if modes != 4:
	print('Modos diferentes a 6, modes:' + str(modes))
	sys.exit()

x = np.zeros([modes, 2*r-1])
for m in range(0 , modes , 2):

	Hx = (S[m , m] * np.array([V[: , m]]).T @ np.array([V[: , m]])
			-  S[m+1 , m+1] * np.array([V[: , m+1]]).T  @ np.array([V[: , m+1]]))

	x[m,:] = Mean_Sum(r, Hx)
	
	Hxx = hankel(x[m,:])[0:r , 0:r]

	Ux, sx, vx = np.linalg.svd(Hxx)
	Sx = np.diag(sx)
	Vx = vx.T
	modesx = Energy_Threshold(sx, 90.0)
	print(m)
	while (modesx / 2.0) > 1:
		print('	1')
				
		i = 0
		Hx = (Sx[0 , 0] * np.array([Vx[: , 0]]).T @ np.array([Vx[: , 0]])
			-  Sx[1 , 1] * np.array([Vx[: , 1]]).T  @ np.array([Vx[: , 1]]))

		x[m,:] = Mean_Sum(r, Hx)

		Hxx = hankel(x[m,:])[0:r , 0:r]

		Ux, sx, vx = np.linalg.svd(Hxx)
		Sx = np.diag(sx)
		Vx = vx.T

		modesx = Energy_Threshold(sx, 90.0)

t1_ievdhm = time.time()


# FIGURE
# fig, ((ax1),
# 	  (ax2),
# 	  (ax3),
# 	  (ax4)) = plt.subplots(4, 1, sharex = True)

# ax1.plot(-x[2,:], dashes = [6,2], label = 's1')
# ax1.plot(s1, label = 's1', color = 'black')
# ax1.set_ylabel('s1')

# ax2.plot(x[4,:], dashes = [6,2], label = 's2')
# ax2.plot(s2, label = 's2', color = 'black')
# ax2.set_ylabel('s2')

# ax3.plot(-x[0,:], label = 's3')
# ax3.plot(s3, label = 's3', color = 'black')
# ax3.set_ylabel('s3')

# xt = x[0,:] + x[2,:] + x[4,:]
# ax4.plot(xt, dashes = [6,2], label = 'st')
# ax4.plot(st, label = 'st', color = 'black')
# ax4.set_xlabel('Time (s)')
# ax4.set_ylabel('st')

# plt.show()


# TEAGER-KAISER ENERGY OPERATOR //////////////////////////////////////////////////////////////
t0_tkeo = time.time()

xi = x.T

xi = np.delete(xi, 3, axis = 1)
xi = np.delete(xi, 1, axis = 1)

N, ns 	= xi.shape
k 		= 15
amp 	= np.zeros([N, ns])
freq 	= np.zeros([N, ns])
dr 		= np.zeros([N, ns])
ce 		= 0

for kk in range(k , N - k):

	x = xi[ce : k + ce , :]

	dx 		= np.zeros([k , ns])
	ex 		= np.zeros([k , ns])
	ex3 	= np.zeros([k , ns])
	ex4 	= np.zeros([k , ns])
	edx 	= np.zeros([k , ns])
	edx3 	= np.zeros([k , ns])

	dx[1:k-1 , :] 	= x[1:k-1 , :] - x[0:k-2 , :]
	dx[0 , :] 		= 2 * dx[1 , :] - dx[2 , :]
	dx[k-1 , :] 	= 2 * dx[k-2 , :] - dx[k-3 , :]

	# TK second-order ...................................................................
	ex[1:k-1 , :] 	= x[1:k-1 , :] * x[1:k-1 , :] - x[2:k , :] * x[0:k-2 , :]
	ex[0 , :] 		= 2 * ex[1 , :] - ex[2 , :]
	ex[k-1 , :] 	= 2 * ex[k-2 , :] - ex[k-3 , :]

	# TK third-order ....................................................................
	ex3[2:k-2 , :] 	= x[2:k-2 , :] * x[3:k-1 , :] - x[4:k , :] * x[1:k-3 , :]
	ex3[0 , :] 		= 2 * ex3[1 , :] - ex3[2 , :]
	ex3[k-1 , :]	= 2 * ex3[k-2 , :] - ex3[k-3 , :]

	# TK second-order ...................................................................
	edx[1:k-1 , :] 	= dx[1:k-1 , :] * dx[1:k-1 , :] - dx[2:k , :] * dx[0:k-2 , :]
	edx[0 , :] 		= 2 * edx[1 , :] - edx[2 , :]
	edx[k-1 , :]	= 2 * edx[k-2 , :] - edx[k-3 , :]

	amp1 			= np.sqrt(abs(ex / ( 1 - (1 - (edx / (2 * ex)))**2 )))
	amp1[0 , :] 	= 2 * amp1[2 , :] - amp1[3 , :]
	amp1[1 , :] 	= 2 * amp1[2 , :] - amp1[3 , :]
	amp1[k-2 , :] 	= 2 * amp1[k-3 , :] - amp1[k-4 , :]
	amp1[k-1 , :] 	= 2 * amp1[k-2 , :] - amp1[k-3 , :]
	for col in range(ns):
		amp[kk-1 , col] 	= np.median(amp1[: , col])

	omega 			= np.zeros([1 , ns])
	omega1 			= np.arccos(1 - (edx / (2 * ex))) * 60.0
	omega1[0 , :] 	= 2 * omega1[2 , :] - omega1[3 , :]
	omega1[1 , :] 	= 2 * omega1[2 , :] - omega1[3 , :]
	omega1[k-2 , :] = 2 * omega1[k-3 , :] - omega1[k-4 , :]
	omega1[k-1 , :] = 2 * omega1[k-2 , :] - omega1[k-3 , :]
	for col in range(ns):
		omega[0 , col] 	= np.median(omega1[: , col])

	damp 			= np.zeros([1 , ns])
	freq[kk-1 , :] 	= omega[0 , :] / (2.0 * np.pi)
	damp2 			= np.log(ex3 / (2 * ex - edx)) * 60.0
	for col in range(ns):
		damp[0 , col] = np.median(damp2[2:k-2 , col])

	dr[kk-1 , :] 	= -100.0 * damp / (np.sqrt(damp**2 + omega**2))
    
	ce = ce + 1

t1_tkeo = time.time()

# # FIGURE FREQUENCY
# fig, ax = plt.subplots(1, 1, sharex = True)
# ax.plot(freq[:,0], label = 's1')
# ax.plot(freq[:,1], label = 's2')
# ax.plot(freq[:,2], label = 's3')
# ax.set(xlabel='Time (s)', ylabel ='y(t)', title='Frequency')
# ax.legend()
# plt.show()

# # FIGURE AMPLITUDE
# fig, ((ax1),
# 	  (ax2),
# 	  (ax3)) = plt.subplots(3, 1, sharex = True)
# ax1.plot(amp[:,0])
# ax1.set_ylabel('s1')
# ax2.plot(amp[:,1])
# ax2.set_ylabel('s2')
# ax3.plot(amp[:,2])
# ax3.set_ylabel('s3')
# ax3.set_xlabel('Time (s)')
# ax1.set(title='Amplitude')
# plt.show()

# # FIGURE DAMPING RATIO
# fig, ax = plt.subplots(1, 1, sharex = True)
# ax.plot(dr[:,0], label = 's1')
# ax.plot(dr[:,1], label = 's2')
# ax.plot(dr[:,2], label = 's3')
# ax.set(xlabel='Time (s)', ylabel ='y(t)', title='Damping Ratio')
# ax.legend()
# plt.show()

print('ievdhm ' + str(t1_ievdhm - t0_ievdhm))
print('tkeo' + str (t1_tkeo - t0_tkeo))