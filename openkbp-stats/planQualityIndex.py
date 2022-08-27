# Compute extra stats for the PMB paper.
# Jiang et al. in Three-dimensional dose prediction for lung IMRT patients with deep neural
# networks: robust learning from heterogeneous beam conﬁgurations compute following metrics:
# PTV: D99, D98, D95, D5 as % of prescribed dose(60Gy) (mean +/- std).
# Eso: D2, V40, V50 (Vx metics as % of volume, Volume receiving dose of atleast x which can be read from DVH plot)
# Cord: D2
# Heart: V35
# Left/Right Lungs: Dmean, V5 and V20
# Total metrics: 15
# Compute all these for all 3 experiments and both ECHO and Manual

import os
import numpy as np
import SimpleITK as sitk
import torch
import argparse
import pandas as pd

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
	oar = data['OAR']
	ptv = data['PTV']
	hist = data['HIST']
	bins = data['BINS']
	hist *= 100
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


def get_dvh(dose, oar, ptv):
	# Compute and return the dvh for all 6 OAR structures
	device = torch.device("cuda:1")
	dose = get_torch_tensor(dose, device)
	oar = get_torch_tensor(oar, device).long()
	oar = torch.nn.functional.one_hot(oar, 6)[..., 1:]  # Remove BG
	oar = oar.permute(3, 0, 1, 2).to(torch.float)
	ptv = get_torch_tensor(ptv, device).long().unsqueeze(dim=0)
	ptv = ptv.to(torch.float)
	oar = torch.cat((oar, ptv), axis=0)

	vols = torch.sum(oar, axis=(1, 2, 3))
	n_bins = 351
	hist = torch.zeros((n_bins, 6)).to(device)
	bins = torch.linspace(0, 70, n_bins)
	bin_w = bins[1] - bins[0]

	for i in range(bins.shape[0]):
		diff = torch.sigmoid((dose - bins[i]) / bin_w)
		diff = torch.cat(6 * [diff.unsqueeze(axis=0)]) * oar
		num = torch.sum(diff, axis=(1, 2, 3))
		hist[i] = (num / vols)*100

	hist_numpy = hist.cpu().numpy()
	bins_np = bins.cpu().numpy()

	return hist_numpy, bins_np


# gt_dir = '../datasets/msk-manual-3d-dvh-beamlet-dense-separate-ptv/test'
# # pred_dir = '../results/unet_128_mae_no_beam_ptv60_dense_manual_separate_ptv/test_latest/npz_images'
# # out_filename = 'MAE_No_Beam_PTV60_Manual.txt'
# #
# # out_dir = './extra_metrics'

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--planName", required=True, help="Enter plan name")
args, _ = parser.parse_known_args()
planName = args.planName
# gt_dir = r'\\pisidsmph\NadeemLab\Gourav\datasets\msk-manual-3d-imrt-sparse-seperate-ptv\test'
gt_dir = '/nadeem_lab/Gourav/datasets/msk-manual-3d-imrt-sparse-seperate-ptv/test'
pred_dir = './results/{}/test_latest/npz_images'.format(planName)
# pred_dir = r'\\pisidsmph\NadeemLab\Gourav\rtdose-predictionNoBeamSaad\results\{}\test_latest\npz_images'.format(planName)
out_filename = '{}_extrastats.txt'.format(planName)
dsc_filename = '{}_dsc.txt'.format(planName)
out_dir = './paper_metrics/{}'.format(planName)
if not os.path.exists(out_dir):
	os.makedirs(out_dir)
metrics_filename = os.path.join(out_dir, out_filename)
dsc_metrics_filename = os.path.join(out_dir, dsc_filename)
# case_file = './resources/test_echo_dose.txt'
case_file = './resources/test_manual_imrt_dose.txt'

cases = get_caselist(case_file)
#cases = cases[:3]

gt_oar_metrics = np.zeros((len(cases), 18), dtype=np.float32)  # 5 OAR metrics (mean) and 3 PTV metrics (D1, D95, D99)
																																# per patient
pred_oar_metrics = np.zeros_like(gt_oar_metrics)
dsc = np.zeros((len(cases), 60), dtype=np.float32)
# OAR indexes; Eso=1, Cord=2, heart=3, Left Lung=4, Right Lung=5
for idx, case in enumerate(cases):
	print('Processing case: {} {} of {}...'.format(case, idx+1, len(cases)))
	gt_dose, oar, ptv, hist, bins = get_gt_case(in_dir=gt_dir, case=case)
	pred_dose = get_pred_case(in_dir=pred_dir, case=case, suffix='_CT2DOSE.nrrd')
	pred_hist, pred_bins = get_dvh(pred_dose, oar, ptv)

	# PTV metrics
	gt_ptv_dose = gt_dose[np.where(ptv==1)]
	pred_ptv_dose = pred_dose[np.where(ptv == 1)]

	# D5, dose received by 5% percent voxels (95th percentile)
	pred_D5 = np.abs((np.percentile(pred_ptv_dose, 95)/60)*100 - (np.percentile(gt_ptv_dose, 95)/60)*100)

	# D95, dose received by 95% percent voxels (5th percentile)
	pred_D95 = np.abs((np.percentile(pred_ptv_dose, 5)/60)*100 - (np.percentile(gt_ptv_dose, 5)/60)*100)

	# D98, dose received by 98% percent voxels (2nd percentile)
	pred_D98 = np.abs((np.percentile(pred_ptv_dose, 2)/60)*100 - (np.percentile(gt_ptv_dose, 2)/60)*100)

	# D99, dose received by 99% percent voxels (1st percentile)
	pred_D99 = np.abs((np.percentile(pred_ptv_dose, 1)/60)*100 - (np.percentile(gt_ptv_dose, 1)/60)*100)

	pred_oar_metrics[idx, 0] = pred_D5
	pred_oar_metrics[idx, 1] = pred_D95
	pred_oar_metrics[idx, 2] = pred_D98
	pred_oar_metrics[idx, 3] = pred_D99

	# OAR metrics
	# Esophagus: D2, V40, V50
	gt_eso_dose = gt_dose[np.where(oar == 1)]
	pred_eso_dose = pred_dose[np.where(oar == 1)]
	eso_D2 = np.abs((np.percentile(pred_eso_dose, 98)/60)*100 - (np.percentile(gt_eso_dose, 98)/60)*100)
	V40_index = int(40/0.2)
	eso_v40 = np.abs(pred_hist[V40_index,0] - hist[V40_index,0])
	V50_index = int(50/0.2)
	eso_v50 = np.abs(pred_hist[V50_index, 0] - hist[V40_index, 0])

	pred_oar_metrics[idx, 4] = eso_D2
	pred_oar_metrics[idx, 5] = eso_v40
	pred_oar_metrics[idx, 6] = eso_v50

	# Cord: D2
	gt_cord_dose = gt_dose[np.where(oar == 2)]
	pred_cord_dose = pred_dose[np.where(oar == 2)]
	cord_D2 = np.abs((np.percentile(pred_cord_dose, 98)/60)*100 - (np.percentile(gt_cord_dose, 98)/60)*100)
	pred_oar_metrics[idx, 7] = cord_D2

	# Heart: V35
	V35_index = int(35 / 0.2)
	heart_v35 = np.abs(pred_hist[V35_index, 3] - hist[V35_index, 3])
	pred_oar_metrics[idx, 8] = heart_v35

	# Left Lung: Dmean, V5, V20
	gt_left_lung_dose = gt_dose[np.where(oar == 4)]
	pred_left_lung_dose = pred_dose[np.where(oar == 4)]
	left_lung_Dmean = np.abs((pred_left_lung_dose.mean()/60)*100 - (gt_left_lung_dose.mean()/60)*100)
	V5_index = int(5 / 0.2)
	left_lung_v5 = np.abs(pred_hist[V5_index, 4] - hist[V5_index, 4])
	V20_index = int(20 / 0.2)
	left_lung_v20 = np.abs(pred_hist[V20_index, 4] - hist[V20_index, 4])

	pred_oar_metrics[idx, 9] = left_lung_Dmean
	pred_oar_metrics[idx, 10] = left_lung_v5
	pred_oar_metrics[idx, 11] = left_lung_v20

	# Right Lung: Dmean, V5, V20
	gt_right_lung_dose = gt_dose[np.where(oar == 5)]
	pred_right_lung_dose = pred_dose[np.where(oar == 5)]
	right_lung_Dmean = np.abs((pred_right_lung_dose.mean() / 60) * 100 - (gt_right_lung_dose.mean() / 60) * 100)
	right_lung_v5 = np.abs(pred_hist[V5_index, 5] - hist[V5_index, 5])
	right_lung_v20 = np.abs(pred_hist[V20_index, 5] - hist[V20_index, 5])

	pred_oar_metrics[idx, 12] = right_lung_Dmean
	pred_oar_metrics[idx, 13] = right_lung_v5
	pred_oar_metrics[idx, 14] = right_lung_v20

	# PTVVol = np.sum(ptv)
	# V60_index = int(60 / 0.2)
	# PTV_V60 = pred_hist[V60_index, 5]

	# calculate homogeneity index
	PTV_D2_pred = np.percentile(pred_ptv_dose, 98)
	PTV_D50_pred = np.percentile(pred_ptv_dose, 50)
	PTV_D98_pred = np.percentile(pred_ptv_dose, 2)
	pred_HI = (PTV_D2_pred - PTV_D98_pred)/PTV_D98_pred

	PTV_D2_gt = np.percentile(gt_ptv_dose, 98)
	PTV_D50_gt = np.percentile(gt_ptv_dose, 50)
	PTV_D98_gt = np.percentile(gt_ptv_dose, 2)
	gt_HI = (PTV_D2_gt - PTV_D98_gt) / PTV_D98_gt
	pred_oar_metrics[idx, 15] = np.abs(gt_HI - pred_HI)

	##calulating paddick conformity index
	pres_iso_dose_mask = (pred_dose >= 60).astype(int)
	V_iso_pres = np.count_nonzero(pres_iso_dose_mask) / 1000
	V_ptv = np.count_nonzero(ptv)/1000
	V_pres_iso_ptv = np.count_nonzero(pres_iso_dose_mask * ptv)/1000
	pred_CI = V_pres_iso_ptv*V_pres_iso_ptv/(V_ptv*V_iso_pres)

	pres_iso_dose_mask = (gt_dose >= 60).astype(int)
	V_iso_pres = np.count_nonzero(pres_iso_dose_mask) / 1000
	V_pres_iso_ptv = np.count_nonzero(pres_iso_dose_mask * ptv) / 1000
	gt_CI = V_pres_iso_ptv * V_pres_iso_ptv / (V_ptv * V_iso_pres)
	pred_oar_metrics[idx, 16] = np.abs(gt_CI - pred_CI)

	pred_oar_metrics[idx, 17] = (np.abs(pred_ptv_dose.mean() - gt_ptv_dose.mean())/60)*100

	##creating dice coefficient
	for i in range(60):
		pred_iso_dose_mask = (pred_dose >= i).astype(int)
		gt_iso_dose_mask = (gt_dose >= i).astype(int)
		V_iso_pred = np.count_nonzero(pred_iso_dose_mask)/1000
		V_iso_gt = np.count_nonzero(gt_iso_dose_mask)/1000

		# pred_iso_dose_i = pred_dose * pred_iso_dose_mask
		# gt_iso_dose_i = gt_dose * gt_iso_dose_mask
		dsc[idx, i] = 2*min(V_iso_pred, V_iso_gt)/(V_iso_pred + V_iso_gt)
	##plot isodose mask and isodose
	# from matplotlib import pyplot as plt
	#
	# plt.rcParams["figure.figsize"] = [7.00, 3.50]
	# plt.rcParams["figure.autolayout"] = True
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# z, x, y = mask.nonzero()
	# ax.scatter(x, y, z, c=z, alpha=1)
	# plt.show()
average = np.mean(pred_oar_metrics, axis=0)
STD = np.std(pred_oar_metrics, axis=0)
average_dsc = np.mean(dsc, axis=0)
STD_dsc = np.std(dsc, axis=0)

with open(metrics_filename, 'w') as f:

	for idx, case in enumerate(cases):
		print_str = '{:<12s}' + '{:<.2f} '*18
		print(print_str.format(case, *pred_oar_metrics[idx]), file=f)

	print('-'*70, file=f)
	print(print_str.format('Average: ', *average), file=f)
	print(print_str.format('STD: ', *STD), file=f)

with open(dsc_metrics_filename, 'w') as f:

	for idx, case in enumerate(cases):
		print_str = '{:<12s}' + '{:<.2f} '*60
		print(print_str.format(case, *dsc[idx]), file=f)

	print('-'*70, file=f)
	print(print_str.format('Average: ', *average_dsc), file=f)
	print(print_str.format('STD: ', *STD_dsc), file=f)
# Gourav
df = pd.read_csv(metrics_filename, delim_whitespace=True, header=None)
# df.columns = ["Cases", "Eso", "Eso", "Eso", "Cord", "Cord", "Cord", "Heart", "Heart", "Heart", "Lung_L", "Lung_L", "Lung_L", "Lung_R", "Lung_R", "Lung_R", "PTV: D1", "PTV: D1","PTV: D1", "PTV: D95", "PTV: D95", "PTV: D95", "PTV: D99", "PTV: D99", "PTV: D99","DVH","MAE"]
df.columns = ["Cases", "PTV:D5", "PTV:D95", "PTV:D98", "PTV:D99", "Eso:D2", "Eso:V40", "Eso:V50", "Cord:D2", "Heart:V35", "Lung_L:Dmean", "Lung_L:V5", "Lung_L:V20", "Lung_R:Dmean", "Lung_R:V5", "Lung_R:V20", "HI", "CI", "PTV:DMean"]
# df.columns = ["Cases", "Eso", "", "", "Cord", "", "", "Heart", "", "", "Lung_L", "", "", "Lung_R", "", "", "PTV: D1", "","", "PTV: D95", "", "", "PTV: D99", "", "", "DVH", "MAE"]
df.to_excel(r"{}.xlsx".format(metrics_filename))

df = pd.read_csv(dsc_metrics_filename, delim_whitespace=True, header=None)
df.to_excel(r"{}.xlsx".format(dsc_metrics_filename))


























