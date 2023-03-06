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
	# rt_dose_itk.SetDirection(dose_img.GetDirection())
		#img_positions.append(ds.ImagePositionPatient[2])
	# x_coord = np.float(dose_img.ImagePositionPatient[0]) + np.float(dose_img.PixelSpacing[0])*np.arange(0, dose_img.Columns)
	# y_coord = np.float(dose_img.ImagePositionPatient[1]) + np.float(dose_img.PixelSpacing[1]) * np.arange(0, dose_img.Rows)
	# z_coord = np.float(dose_img.ImagePositionPatient[2]) + np.array(dose_img.GridFrameOffsetVector)
	# dose_array = np.zeros(sitk.GetArrayFromImage(arr_dose).shape, dtype=np.float)

	# indexes = np.argsort(np.asarray(img_positions))
	# dicom_names = list(np.asarray(dicom_paths)[indexes])
	#
	#reader = sitk.ImageSeriesReader()
	# reader.SetFileNames(dicom_names)
	# img = reader.Execute()

	# return dose_img
	return rt_dose_itk

# in_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO_dicomset'
# out_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/pCT_Dose_ECHO_dicomset_extracted'

in_dir = r'\\pensmph6\MpcsResearch1\YangJ\lung-echo\out'
# out_dir = r'\\pisidsmph\NadeemLab\Gourav\lung-echo-dicomextracted'
out_dir = r'\\pisidsmph\NadeemLab\Gourav\lung-manual-dicomextracted'
cases = ['10000000']
# in_dir = r'\\pensmph6\mpcsresearch1\YangJ\lung-echo\test1'
# out_dir = r'\\pensmph6\mpcsresearch1\YangJ\lung-echo\Pred'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

cases = os.listdir(in_dir)

for idx, case in enumerate(cases):
	try:
		print('Processing case {}: {} of {} ...'.format(case, idx + 1, len(cases)))
		dose_img = read_dicom(in_dir, case)
		filename = os.path.join(out_dir, case + '_dose.nrrd')
		# filename = os.path.join(out_dir, case + '_dose.nrrd')
		# dose_img.save_as(filename)
		sitk.WriteImage(dose_img, filename)
	except:
		print('Processing of case {} failed'.format(case))
		pass
