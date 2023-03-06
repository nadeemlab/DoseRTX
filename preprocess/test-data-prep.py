# Test if 3D data with DVH histograms are properly prepared

import os
import numpy as np
import matplotlib.pyplot as plt


in_dir = r'\\pisidsmph\NadeemLab\Interns\Navdeep\msk-echo-3d-dvh-beamlet-sparse-separate-ptv\test'

case = 'LUNG1-002'
filename = os.path.join(in_dir, case)
data = np.load(filename)

ct = data['CT']
dose = data['DOSE']
oar = data['OAR']
hist = data['HIST']
bins = data['BINS']
beam = data['BEAM']

print(hist.shape)
print(hist[:,2])
# plt.imshow(beam[65], cmap='gray')
# plt.show()
#
# plt.imshow(dose[65], cmap='gray')
# plt.show()
fig = plt.figure()
plt.plot(hist[:,2])
plt.show()
for i in range(hist.shape[1]):
	plt.plot(bins, hist[:,i])

plt.legend(['Eso', 'Cord', 'Heart', 'Lung_L', 'Lung_R', 'PTV'])
plt.xlabel('Dose (Gy)')
plt.ylabel('Volume Fraction (%)')
plt.show()