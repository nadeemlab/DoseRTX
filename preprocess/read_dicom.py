# Read dicom files from dicom directory
import os
import SimpleITK as sitk
from pydicom import dcmread
import numpy as np


def read_dicom(in_dir, case):
	dicom_names = os.listdir(os.path.join(in_dir, case))
	dicom_paths = []
	for dcm in dicom_names:
		if dcm[:2] == 'CT':
			dicom_paths.append(os.path.join(in_dir, case, dcm))

	img_positions = []
	for dcm in dicom_paths:
		ds = dcmread(dcm)
		img_positions.append(ds.ImagePositionPatient[2])

	indexes = np.argsort(np.asarray(img_positions))
	dicom_names = list(np.asarray(dicom_paths)[indexes])

	reader = sitk.ImageSeriesReader()
	reader.SetFileNames(dicom_names)
	img = reader.Execute()

	return img


# in_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO_dicomset'
# out_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO_dicomset_extracted'

in_dir = r'\\pensmph6\MpcsResearch1\YangJ\lung-echo\out-ECHO'
out_dir = r'\\pisidsmph\NadeemLab\Gourav\lung-echo-dicomextracted'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

cases = os.listdir(in_dir)

for idx, case in enumerate(cases):
	print('Processing case {}: {} of {} ...'.format(case, idx+1, len(cases)))
	img = read_dicom(in_dir, case)
	filename = os.path.join(out_dir, case + '_CT.nrrd')
	sitk.WriteImage(img, filename)
