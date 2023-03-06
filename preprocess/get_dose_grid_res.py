# The dense interpolation of un-modulated code provided by Gourav requires the 3d grid resolution of the corresponding
# echo dose (or manual dose). Get grid resolution of all 120 cases and save in text files.

import os
import SimpleITK as sitk


def get_caselist(txt_file):
	datasets = []
	with open(txt_file, 'r') as f:
		for dset in f:
			datasets.append(dset.strip())
	return datasets


in_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO'
out_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/dose_beamlet_3D_martices/influenceMatrix_beamlet_info'
train_cases = get_caselist('../resources/train_echo_dose.txt')
test_cases = get_caselist('../resources/test_echo_dose.txt')

all_cases = train_cases + test_cases

for idx, case in enumerate(all_cases):
	print('Processing case {}: {} of {} ...'.format(case, idx+1, len(all_cases)))
	#filename = os.path.join(in_dir, case + '_dose_ECHO.nrrd')
	filename = os.path.join(in_dir, case + '_dose.nrrd')
	dose = sitk.ReadImage(filename)
	grid_res = dose.GetSpacing()

	out_file = os.path.join(out_dir, case , 'grid_res_manual.txt')
	with(open(out_file, 'w')) as f:
		print(grid_res[0], grid_res[1], grid_res[2], file=f)