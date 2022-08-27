# Compute predicticted dose quality statistics
import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from similarity_metrics import mse, rmse, ssim, mae


def get_caselist(txt_file):
	datasets = []
	with open(txt_file, 'r') as f:
		for dset in f:
			datasets.append(dset.strip())
	return datasets


def get_gt_case(in_dir, case):
	filename = os.path.join(in_dir, case + '.npz')
	data = np.load(filename)
	dose = data['DOSE']

	return dose


def get_pred_case(in_dir, case, suffix):
	filename = os.path.join(in_dir, case + suffix)
	dose = sitk.ReadImage(filename)
	dose = sitk.GetArrayFromImage(dose)
	return dose


#gt_dir = '../datasets/msk-echo-3d-dvh-better-cropped/test'
gt_dir = '../datasets/msk-manual-3d-dvh-beamlet-dense/test'

pred_dir = '../results/unet_128_mae_beamlet_dense_manual/test_latest/npz_images'
metrics_filename = 'Unet_128_MAE_Beamlet_Dense_Manual.txt'
case_file = '../resources/test_echo_dose.txt'

cases = get_caselist(case_file)

metrics = np.zeros((len(cases), 4), dtype=np.float32)

for idx, case in enumerate(cases):
	gt_dose = get_gt_case(in_dir=gt_dir, case=case)
	pred_dose = get_pred_case(in_dir=pred_dir, case=case, suffix='_CT2DOSE.nrrd')

	SSIM = ssim(gt_dose/70, pred_dose/70)  # GT dose was clipped to 0 -- 70

	MAE = mae(gt_dose, pred_dose)
	RMSE = rmse(gt_dose, pred_dose)
	MSE = mse(gt_dose, pred_dose)

	metrics[idx] = np.asarray([SSIM, MAE, MSE, RMSE], dtype=np.float32)

# Print metrics to file
ind = np.argsort(metrics[:,1])  # Sort by MAE
metrics = metrics[ind]
cases = list(np.asarray(cases)[ind])

metrics_file = open(metrics_filename, 'w')
metric_headings = ['Case', 'SSIM', 'MAE', 'MSE', 'RMSE']
heading = '{:<12s}'
print_str = '{:<12s}'
for i in range(4):
	heading += ' {:<4s} '
	print_str += ' {:<.2f} '
print(heading.format(*metric_headings), file=metrics_file)
print("-"*40, file=metrics_file)

for i, case in enumerate(cases):
	print(print_str.format(case, *metrics[i]), file=metrics_file)

print("-"*40, file=metrics_file)
avg  = np.mean(metrics, axis=0)
minm = np.min(metrics, axis=0)
maxm = np.max(metrics, axis=0)
stddev  = np.std(metrics, axis=0)
print(print_str.format('average', *avg), file=metrics_file)
print(print_str.format('Std:', *stddev), file=metrics_file)
print(print_str.format('minimum', *minm), file=metrics_file)
print(print_str.format('maximum', *maxm), file=metrics_file)

metrics_file.close()
