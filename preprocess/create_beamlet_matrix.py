# Read the influence matrix from text files sent by Gourav
# Format is x y z value one per line
# x,y,z seem to be in physical units. So assuming we can use the echo dose files header info to map physical to
# grid indexes.
# Normalize the beamlet to have same amx min values as the corresponding echo dose

import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import matplotlib.pyplot as plt


def get_caselist(txt_file):
	datasets = []
	with open(txt_file, 'r') as f:
		for dset in f:
			datasets.append(dset.strip())
	return datasets


def get_dataset(in_dir, case, suffix, itk=False):
	filename = os.path.join(in_dir, case + suffix)
	img = sitk.ReadImage(filename)
	if itk is False:
		img = sitk.GetArrayFromImage(img)

	return img


def read_influence_matrix(in_dir, case):
	filename = os.path.join(in_dir, case, 'Dose.txt')  # Sparse
	# filename = os.path.join(in_dir, case, 'Dose1.txt')   # Dense Beam Echo
	# filename = os.path.join(in_dir, case, 'Dose2.txt')  # Dense beam Manual
	beamlet_info = np.genfromtxt(filename)

	return beamlet_info


in_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO'
beamlet_info_in_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/dose_beamlet_3D_martices/influenceMatrix_beamlet_info'
out_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO'
# case_file = '../resources/train_echo_dose.txt'
case_file = '../resources/test_echo_dose.txt'
cases = get_caselist(case_file)

normalize = True

for idx, case in enumerate(cases):

	print('Processing case: {} {} of {} ...'.format(case, idx+1, len(cases)))
	#dose = get_dataset(in_dir, case, '_dose_ECHO.nrrd', itk=True)
	dose = get_dataset(in_dir, case, '_dose.nrrd', itk=True)  # Manual dose
	beamlet_info = read_influence_matrix(beamlet_info_in_dir, case)
	beamlet_arr = np.zeros(sitk.GetArrayFromImage(dose).shape, dtype=np.float)

	for row in beamlet_info:
		curr_pt = (row[0], row[1], row[2])
		curr_val = row[3]

		curr_indx = dose.TransformPhysicalPointToIndex(curr_pt)  # X,Y,Z

		beamlet_arr[curr_indx[2], curr_indx[1], curr_indx[0]] = curr_val

	if normalize is True:
		dose_arr = sitk.GetArrayFromImage(dose)
		tmaxm = dose_arr.max()
		tminm = dose_arr.min()

		rmin = beamlet_arr.min()
		rmax = beamlet_arr.max()
		rrange = rmax - rmin

		if rrange > 0:
			beamlet_arr = ((beamlet_arr - rmin) / rrange) * (tmaxm - tminm) + tminm  # Beamlet array normalized to have same range as dose array
		else:
			print('\tBeamlet all zeros.')

	beamlet_itk = sitk.GetImageFromArray(beamlet_arr)
	beamlet_itk.SetOrigin(dose.GetOrigin())
	beamlet_itk.SetSpacing(dose.GetSpacing())
	beamlet_itk.SetDirection(dose.GetDirection())

	# filename = os.path.join(out_dir, case + '_echo_dose_beamlet.nrrd')
	# filename = os.path.join(out_dir, case + '_manual_dose_beamlet.nrrd')
	# filename = os.path.join(out_dir, case + '_echo_dose_beamlet_sparse.nrrd')
	filename = os.path.join(out_dir, case + '_manual_dose_beamlet_sparse.nrrd')
	sitk.WriteImage(beamlet_itk, filename)


