# Compute dose volume histograms of Ground Truth and Predicted dose for selected case
import os
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
import random
import argparse

def get_caselist(txt_file):
	datasets = []
	with open(txt_file, 'r') as f:
		for dset in f:
			datasets.append(dset.strip())
	return datasets


def get_colors(num):
	random.seed(42)
	colors = []
	for i in range(num):
		color = (random.random(), random.random(), random.random())
		colors.append(color)

	return colors


def get_gt_case(in_dir, case):
	filename = os.path.join(in_dir, case + '.npz')
	data = np.load(filename)
	dose = data['DOSE']
	oar = data['OAR']
	ptv = data['PTV']
	hist = data['HIST']
	bins = data['BINS']

	return dose, oar, ptv, hist, bins


def get_pred_case(in_dir, case, suffix):
	filename = os.path.join(in_dir, case + suffix)
	dose = sitk.ReadImage(filename)
	dose = sitk.GetArrayFromImage(dose)

	return dose


def get_torch_tensor(npy_tensor, device):
	out = torch.from_numpy(npy_tensor)
	out.to(device)

	return out

def get_dvh(dose, oar, ptv, bins):
	# Compute and return the dvh for all 6 OAR structures
	device = torch.device('cuda:0')
	dose = get_torch_tensor(dose, device)
	oar = get_torch_tensor(oar, device).long()
	oar = torch.nn.functional.one_hot(oar, 6)[..., 1:]  # Remove BG
	oar = oar.permute(3, 0, 1, 2).to(torch.float)
	ptv = get_torch_tensor(ptv, device).long().unsqueeze(dim=0)
	ptv = ptv.to(torch.float)
	oar = torch.cat((oar, ptv), axis=0)
	bins = get_torch_tensor(bins, device)
	vols = torch.sum(oar, axis=(1, 2, 3))
	n_bins = bins.shape[0]
	hist = torch.zeros((n_bins, 6)).to(device)
	# bins = torch.linspace(0, 70, n_bins)
	bin_w = bins[1] - bins[0]

	for i in range(bins.shape[0]):
		diff = torch.sigmoid((dose - bins[i]) / bin_w)
		diff = torch.cat(6 * [diff.unsqueeze(axis=0)]) * oar
		num = torch.sum(diff, axis=(1, 2, 3))
		hist[i] = (num / vols)*100

	hist_numpy = hist.cpu().numpy()
	#bins_np = bins.cpu().numpy()

	return hist_numpy

# def get_dvh(dose, oar, bins):
# 	# Compute and return the dvh for all 6 OAR structures
# 	# bins used for cumputing GT histogram
#
# 	device = torch.device('cuda:0')
# 	dose = get_torch_tensor(dose, device)
# 	oar = get_torch_tensor(oar, device).long()
# 	oar = torch.nn.functional.one_hot(oar, 7)[..., 1:]  # Remove BG
# 	oar = oar.permute(3, 0, 1, 2).to(torch.float)
# 	bins = get_torch_tensor(bins, device)
#
# 	vols = torch.sum(oar, axis=(1, 2, 3))
# 	n_bins = bins.shape[0]
# 	hist = torch.zeros((n_bins, 6)).to(device)
# 	bin_w = bins[1] - bins[0]
#
# 	for i in range(bins.shape[0]):
# 		diff = torch.sigmoid((dose - bins[i]) / bin_w)
# 		diff = torch.cat(6 * [diff.unsqueeze(axis=0)]) * oar
# 		num = torch.sum(diff, axis=(1, 2, 3))
# 		hist[i] = (num / vols) * 100
#
# 	hist_numpy = hist.cpu().numpy()
#
# 	return hist_numpy

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("--planName", required=True, help="Enter plan name")
args, _ = parser.parse_known_args()
planName = args.planName
# gt_dir = '/nadeem_lab/Gourav/datasets/msk-manual-3d-imrt-sparse-seperate-ptv-256_256_128/test'
gt_dir = '/nadeem_lab/Gourav/datasets/msk-manual-3d-imrt-sparse-seperate-ptv/test'
pred_dir = './results/{}/test_latest/npz_images'.format(planName)
out_dir = './results/{}/test_latest/DVHPlots'.format(planName)
if not os.path.exists(out_dir):
	os.makedirs(out_dir)
# case_file = './resources/test_echo_dose.txt'
case_file = './resources/test_manual_imrt_dose.txt'
cases = get_caselist(case_file)
colors = get_colors(6)

for idx, case in enumerate(cases):
	print('Processing case: {} {} of {}...'.format(case, idx + 1, len(cases)))
	dose, oar, ptv, hist, bins = get_gt_case(gt_dir, case)
	hist *= 100
	pred_dose = get_pred_case(pred_dir, case, '_CT2DOSE.nrrd')
	pred_hist = get_dvh(pred_dose, oar, ptv, bins)

	legends = ['Cord', 'Eso', 'Heart', 'Lung_L', 'Lung_R', 'PTV']
	for i in range(hist.shape[1]):
		plt.plot(bins, hist[:,i], color=colors[i],linestyle='solid', label=legends[i])
		plt.plot(bins, pred_hist[:,i], color=colors[i], linestyle='dashed', label=None)
	plt.legend(loc='best')
	plt.suptitle('Case ' + case)
	plt.title('Dashed: Predicted')
	plt.xlabel('Dose (Gy)')
	plt.ylabel('Volume Fraction (%)')
	plt.savefig(os.path.join(out_dir, case + '.png'))
	#plt.show()
	plt.close()


