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
