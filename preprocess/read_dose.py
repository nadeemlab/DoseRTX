# Read dicom files from dicom directory
import os
import SimpleITK as sitk
from pydicom import dcmread
import numpy as np


def read_dicom(in_dir, case):
	dicom_names = os.listdir(os.path.join(in_dir, case))
	dicom_paths = []
	for dcm in dicom_names:
		if dcm[:2] == 'RD':
			dose_file_name = os.path.join(in_dir, case, dcm)

	img_positions = []
	#for dcm in dicom_paths:
	#	ds = dcmread(dcm)
	# img = ds.pixel_array
	dose_img = dcmread(dose_file_name)
	# dose_img = sitk.ReadImage([dcm for dcm in dicom_paths])
	arr_dose = dose_img.pixel_array
	rt_dose = arr_dose*dose_img.DoseGridScaling
	rt_dose_itk = sitk.GetImageFromArray(rt_dose)
	rt_dose_itk.SetOrigin(dose_img.ImagePositionPatient)
	rt_dose_itk.SetSpacing([np.float(dose_img.PixelSpacing[0]), np.float(dose_img.PixelSpacing[1]), dose_img.GridFrameOffsetVector[1]-dose_img.GridFrameOffsetVector[0]])
	return rt_dose_itk
