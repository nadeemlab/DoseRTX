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
pred_dir_dvh = './results/manualNoBeam290Patients128_128_128_DVHLossStandUNET/test_latest/npz_images'
pred_dir_mom = './results/manualNoBeam290Patients128_128_128_MomLossStandUNETW0_05_run2/test_latest/npz_images'
pred_dir_mae = './results/manualNoBeam290Patients128_128_128_MAELossStandUNET/test_latest/npz_images'
out_dir = './results/{}/test_latest/DVHPlots5'.format(planName)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# case_file = './resources/test_echo_dose.txt'
case_file = './resources/test_manual.txt'
colors = get_colors(6)

for idx, case in enumerate(cases):
    print('Processing case: {} {} of {}...'.format(case, idx + 1, len(cases)))
    dose, oar, ptv, hist, bins = get_gt_case(gt_dir, case)
    hist *= 100
    pred_dose = get_pred_case(pred_dir_mae, case, '_CT2DOSE.nrrd')
    pred_hist_mae = get_dvh(pred_dose, oar, ptv, bins)
    pred_dose_dvh = get_pred_case(pred_dir_dvh, case, '_CT2DOSE.nrrd')
    pred_hist_dvh = get_dvh(pred_dose_dvh, oar, ptv, bins)
    pred_dose_mom = get_pred_case(pred_dir_mom, case, '_CT2DOSE.nrrd')
    pred_hist_mom = get_dvh(pred_dose_mom, oar, ptv, bins)

    legends = ['Cord', 'Esophagus', 'Heart', 'Lung_L', 'Lung_R', 'PTV']
    plt.rcParams['font.size'] = '8'
    # plt.subplots_adjust(right=0.85)
    fig = plt.figure(figsize=(6.1667, 5.1667))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    label = ['Actual', 'MAE loss', 'MAE+DVH loss', 'MAE+Moment loss']
    for i in range(hist.shape[1]):
        # if i == 1:
            # plt.plot(bins, hist[:,i], color=colors[i],linestyle='solid', label=legends[i])
            # # plt.plot(bins, pred_hist_mae[:,i], color=colors[i], linestyle='dashed', label=None)
            # plt.plot(bins, pred_hist_dvh[:, i], color=colors[i], linestyle='dashed', label=None)
            # plt.plot(bins, pred_hist_mom[:, i], color=colors[i], linestyle='dotted', label=None)
        # plt.rcParams['font.size'] = '16'

        # plt.plot(bins, hist[:,i], color=colors[0], label='Actual')
        # plt.plot(bins, pred_hist_mae[:,i], color=colors[1], label='MAE loss')
        # plt.plot(bins, pred_hist_dvh[:, i], color=colors[2], label='MAE+DVH loss')
        # plt.plot(bins, pred_hist_mom[:, i], color=colors[3], label='MAE+Moment loss')
        # plt.legend(loc='best')
        # plt.suptitle('Case ' + case)
        # ## plt.title('Dashed: MAE Predicted, Dash-Dot: DVH Predicted, Dot: Mom Predicted')
        # # plt.title('Solid:Actual, Dashed: DVH Predicted, Dot: Moment Predicted')
        # plt.title(legends[i])
        # plt.xlabel('Dose (Gy)')
        # plt.ylabel('Volume Fraction (%)')
        # plt.savefig(os.path.join(out_dir, case + legends[i] + '.png'))
        # #plt.show()
        # plt.close()

        ax = fig.add_subplot(2, 3, i+1)
        if i == hist.shape[1]-1:
            ax.plot(bins[-150:], hist[-150:, i], label='Actual', linewidth=1, linestyle='solid', color=colors[i])
            # ax.plot(bins, pred_hist_mae[:, i], label='MAE loss', linewidth=1)
            ax.plot(bins[-150:], pred_hist_dvh[-150:, i], label='(MAE+DVH) loss', linewidth=1, linestyle='dashed', color=colors[i])
            ax.plot(bins[-150:], pred_hist_mom[-150:, i], label='(MAE+Moment) loss', linewidth=1, linestyle='dotted', color=colors[i])
            ax.set_xticks(np.arange(min(bins[-150:]), max(bins[-150:]) + 1, 10))
        else:
            ax.plot(bins, hist[:, i], label='Actual', linewidth=1, linestyle='solid', color=colors[i])
            # ax.plot(bins, pred_hist_mae[:, i], label='MAE loss', linewidth=1)
            ax.plot(bins, pred_hist_dvh[:, i], label='(MAE+DVH) loss', linewidth=1, linestyle='dashed', color=colors[i])
            ax.plot(bins, pred_hist_mom[:, i], label='(MAE+Moment) loss', linewidth=1, linestyle='dotted',
                    color=colors[i])
            ax.set_xticks(np.arange(min(bins), max(bins) + 1, 20))
            # ax.legend(loc='best')
            # ax.suptitle('Case ' + case)
            ## plt.title('Dashed: MAE Predicted, Dash-Dot: DVH Predicted, Dot: Mom Predicted')
            # plt.title('Solid:Actual, Dashed: DVH Predicted, Dot: Moment Predicted')
        ax.set_title(legends[i])
        ax.set(xlabel='Dose (Gy)', ylabel='Volume Fraction (%)')
            # ax.xlabel('Dose (Gy)')
            # ax.ylabel('Volume Fraction (%)')

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    lg = fig.legend(lines, labels, loc='lower right', prop={'size': 8}, bbox_to_anchor=(1.02, 1.0))
    # lgd = fig.legend(label, loc="center right", prop={'size': 6}, bbox_to_anchor=(0.9, 0.5))
    fig.tight_layout()
    # fig.subplots_adjust(top=0.80)
    fig.savefig(os.path.join(out_dir, case + '.png'), dpi=300, bbox_extra_artists=(lg,), bbox_inches='tight')




