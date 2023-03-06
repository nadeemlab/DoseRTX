# Dose and ct files have different dimensions. Match the FOV of dose/ct to create expanded dose array which matches
# CT so that they can be used for training.

import os
import numpy as np
import SimpleITK as sitk


def get_caselist(txt_file):
	datasets = []
	with open(txt_file, 'r') as f:
		for dset in f:
			datasets.append(dset.strip())
	return datasets


def resample(img, ref_image):
	resampler = sitk.ResampleImageFilter()
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetReferenceImage(ref_image)
	img = resampler.Execute(img)
	
	return img


def resample_cases(case_file, in_dir, only_beamlet=True):

	cases = get_caselist(case_file)
	ct_exists, echo_dose_beamlet_exists, dose_exists, echo_dose_exists = False, False, False, False
	for idx, case in enumerate(cases):
		print('Processing case {}: {} of {} ...'.format(case, idx+1, len(cases)))
		filename = os.path.join(in_dir, case + '_CT.nrrd')
		if os.path.exists(filename):
			ct_exists = True
			ct = sitk.ReadImage(filename)

		#filename = os.path.join(in_dir, case + '_echo_dose_beamlet.nrrd')
		# filename = os.path.join(in_dir, case + '_manual_dose_beamlet.nrrd')  # Manual dose beamlet dense
		# filename = os.path.join(in_dir, case + '_manual_dose_beamlet_sparse.nrrd')  # Manual dose sparse beamlet
		filename = os.path.join(in_dir, case + '_echo_dose_beamlet_sparse.nrrd')  # Manual dose sparse beamlet
		if os.path.exists(filename):
			echo_dose_beamlet_exists = True
			echo_dose_beamlet = sitk.ReadImage(filename)

		if only_beamlet is False:
			filename = os.path.join(in_dir, case + '_dose.nrrd')
			if os.path.exists(filename):
				dose_exists = True
				dose = sitk.ReadImage(filename)

			filename = os.path.join(in_dir, case + '_dose_ECHO.nrrd')
			if os.path.exists(filename):
				echo_dose_exists = True
				echo_dose = sitk.ReadImage(filename)

		if ct_exists:
			if only_beamlet is False:
				if dose_exists:
					dose_resampled = resample(dose, ct)
					filename = os.path.join(in_dir, case + '_dose_resampled.nrrd')
					sitk.WriteImage(dose_resampled, filename)
				if echo_dose_exists:
					echo_dose_resampled = resample(echo_dose, ct)
					filename = os.path.join(in_dir, case + '_echo_dose_resampled.nrrd')
					sitk.WriteImage(echo_dose_resampled, filename)

			if echo_dose_beamlet_exists:
				dose_beamlet_resampled = resample(echo_dose_beamlet, ct)
				#filename = os.path.join(in_dir, case + '_echo_dose_beamlet_resampled.nrrd')
				# filename = os.path.join(in_dir, case + '_manual_dose_beamlet_resampled.nrrd')
				filename = os.path.join(in_dir, case + '_manual_dose_beamlet_sparse_resampled.nrrd')
				# filename = os.path.join(in_dir, case + '_echo_dose_beamlet_sparse_resampled.nrrd')
				sitk.WriteImage(dose_beamlet_resampled, filename)

		# print(ct.GetPixelIDTypeAsString(), dose_resampled.GetPixelIDTypeAsString(),dose.GetPixelIDTypeAsString())
		# print(ct.GetSize(), dose_resampled.GetSize())
		# print(ct.GetOrigin(), dose_resampled.GetOrigin())
		# print(ct.GetSpacing(), dose_resampled.GetSpacing())
		# print(ct.GetDirection(), dose_resampled.GetDirection())
		# print(sitk.GetArrayFromImage(dose).min(), sitk.GetArrayFromImage(dose).max())
		# print(sitk.GetArrayFromImage(dose_resampled).min(), sitk.GetArrayFromImage(dose_resampled).max())

in_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO'

# case_file = '../resources/train_echo_dose.txt'
# resample_cases(case_file, in_dir, only_beamlet=True)

case_file = '../resources/test_echo_dose.txt'
resample_cases(case_file, in_dir, only_beamlet=True)
