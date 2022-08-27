# Computet the DVH stats used by Open KBP paper.
# For each OAR compute mean dose. Paper also computes "the maximum dose received by 0.1cc of OAR i". Need to figure out
# how to compute that.
# OAR contains labels Eso = 1, Cord = 2, Heart = 3, Left Lung = 4, Right Lung = 5, PTV = 6

import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import argparse


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
    return dose, oar, ptv


def get_pred_case(in_dir, case, suffix):
    filename = os.path.join(in_dir, case + suffix)
    dose = sitk.ReadImage(filename)
    dose = sitk.GetArrayFromImage(dose)
    return dose


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("--planName", required=True, help="Enter plan name")
args, _ = parser.parse_known_args()
planName = args.planName
gt_dir = '/nadeem_lab/Gourav/datasets/msk-manual-3d-imrt-sparse-seperate-ptv/test'
pred_dir = './results/{}/test_latest/npz_images'.format(planName)

out_filename = '{}.txt'.format(planName)

out_dir = './paper_metrics/{}'.format(planName)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
metrics_filename = os.path.join(out_dir, out_filename)

# case_file = './resources/test_echo_dose.txt'
# case_file = './resources/test_manual_dose.txt'
case_file = './resources/test_manual_imrt_dose.txt'
cases = get_caselist(case_file)
# cases = cases[:1]

gt_oar_metrics = np.zeros((len(cases), 5 + 3),
                          dtype=np.float32)  # 5 OAR metrics (mean) and 3 PTV metrics (D1, D95, D99)
# per patient
pred_oar_metrics = np.zeros_like(gt_oar_metrics)

gt_oar_metrics_max = np.zeros((len(cases), 5),
                              dtype=np.float32)  # 5 OAR metrics (mean) and 3 PTV metrics (D1, D95, D99)
# per patient
pred_oar_metrics_max = np.zeros_like(gt_oar_metrics_max)

##D(0.1cc)
mae_metrics = np.zeros((len(cases)), dtype=np.float32)
mae_metrics_oar = np.zeros((len(cases), 5 + 3), dtype=np.float32)

for idx, case in enumerate(cases):
    print('Processing case: {} {} of {}...'.format(case, idx + 1, len(cases)))
    gt_dose, oar, ptv = get_gt_case(in_dir=gt_dir, case=case)
    pred_dose = get_pred_case(in_dir=pred_dir, case=case, suffix='_CT2DOSE.nrrd')

    dose_error = np.mean(np.abs(gt_dose - pred_dose))
    mae_metrics[idx] = dose_error

    # DVH Metrics
    # OAR metrics (For now mean dose received by OAR)
    for i in range(1, 6):
        gt_oar_dose = gt_dose[np.where(oar == i)]
        pred_oar_dose = pred_dose[np.where(oar == i)]

        gt_oar_metrics[idx, i - 1] = gt_oar_dose.mean()
        pred_oar_metrics[idx, i - 1] = pred_oar_dose.mean()
        gt_oar_metrics_max[idx, i - 1] = gt_oar_dose.max()
        pred_oar_metrics_max[idx, i - 1] = pred_oar_dose.max()
        gt_oar_metrics_max[idx, i - 1] = gt_oar_dose.max()
        pred_oar_metrics_max[idx, i - 1] = pred_oar_dose.max()

    gt_ptv_dose = gt_dose[np.where(ptv == 1)]
    pred_ptv_dose = pred_dose[np.where(ptv == 1)]

    # D1, dose received by 1% percent voxels (99th percentile)
    gt_D1 = np.percentile(gt_ptv_dose, 99)
    pred_D1 = np.percentile(pred_ptv_dose, 99)

    # D95, dose received by 95% percent voxels (5th percentile)
    gt_D95 = np.percentile(gt_ptv_dose, 5)
    pred_D95 = np.percentile(pred_ptv_dose, 5)

    # D99, dose received by 99% percent voxels (1st percentile)
    gt_D99 = np.percentile(gt_ptv_dose, 1)
    pred_D99 = np.percentile(pred_ptv_dose, 1)

    gt_oar_metrics[idx, 5] = gt_D1
    gt_oar_metrics[idx, 6] = gt_D95
    gt_oar_metrics[idx, 7] = gt_D99

    pred_oar_metrics[idx, 5] = pred_D1
    pred_oar_metrics[idx, 6] = pred_D95
    pred_oar_metrics[idx, 7] = pred_D99

dvh_errors = np.abs(gt_oar_metrics - pred_oar_metrics)
dvh_errors_max = np.abs(gt_oar_metrics_max - pred_oar_metrics_max)
dvh_errors_all = np.append(dvh_errors, dvh_errors_max, 1)
dvh_errors_dset_mean = np.mean(dvh_errors_all, axis=1)  # Mean of 8 DVH errors for each dataset
print(dvh_errors_all.mean())
print(*dvh_errors_dset_mean)

with open(metrics_filename, 'w') as f:
    # Print headings
    first_heading = '{:<8s}' + '{: ^17s}' * 5 + '{: ^54s}'
    print(first_heading.format(' ', 'Cord', 'Eso', 'Heart', 'Lung_L', 'Lung_R', 'PTV: D1, D95, D99'), file=f)
    # second_heading = '{:<8s}' + '{: ^5s}'*3 + '{: ^5s}'*3 + '{: ^5s}'*3 + '{: ^5s}'*3 + '{: ^5s}'*3 + '{: ^5s}'*3 + '{: ^5s}'*3 + '{: ^5s}'*3
    second_heading = 'Case     ' + 'GT(Mean)   Pred(Mean)  Err(Mean)   GT(Max)   Pred(Max)  Err(Max) ' * 5 + \
                     'GT   Pred  Err ' * 3 + 'DVH ' + 'MAE'
    print(second_heading, file=f)
    for idx, case in enumerate(cases):
        print_str = '{} '.format(case)
        # curr_metric = '{:<.2f} '*3  # Gt, pred, mae(gt, pred)
        curr_metric_oar = '{:<.2f} ' * 6  # Above and max dose
        curr_metric_ptv = '{:<.2f} ' * 3
        for i in range(8):
            if i < 5:
                print_str += curr_metric_oar.format(gt_oar_metrics[idx, i], pred_oar_metrics[idx, i],
                                                    dvh_errors[idx, i], gt_oar_metrics_max[idx, i],
                                                    pred_oar_metrics_max[idx, i], dvh_errors_max[idx, i])
            else:
                print_str += curr_metric_ptv.format(gt_oar_metrics[idx, i], pred_oar_metrics[idx, i],
                                                    dvh_errors[idx, i])

        print_str += '{:<.2f} '.format(dvh_errors_dset_mean[idx])
        print_str += '{:<.2f}'.format(mae_metrics[idx])

        print(print_str, file=f)

    print('-' * 70, file=f)
    print('DVH Score: {:<.2f}'.format(dvh_errors_all.mean()), file=f)
    print('Dose Score: {:<.2f}'.format(mae_metrics.mean()), file=f)

# Gourav
df = pd.read_csv(metrics_filename, delim_whitespace=True, header=None, skiprows=1)
# df.columns = ["Cases", "Eso", "Eso", "Eso", "Cord", "Cord", "Cord", "Heart", "Heart", "Heart", "Lung_L", "Lung_L", "Lung_L", "Lung_R", "Lung_R", "Lung_R", "PTV: D1", "PTV: D1","PTV: D1", "PTV: D95", "PTV: D95", "PTV: D95", "PTV: D99", "PTV: D99", "PTV: D99","DVH","MAE"]
df.columns = ["Cases", "Cord", "", "", "", "", "", "Eso", "", "", "", "", "", "Heart", "", "", "", "", "", "Lung_L", "",
              "", "", "", "", "Lung_R", "", "", "", "", "", "PTV: D1", "", "", "PTV: D95", "", "", "PTV: D99", "", "",
              "DVH", "MAE"]
# df.columns = ["Cases", "Eso", "", "", "Cord", "", "", "Heart", "", "", "Lung_L", "", "", "Lung_R", "", "", "PTV: D1", "","", "PTV: D95", "", "", "PTV: D99", "", "", "DVH", "MAE"]
df.to_excel(r"{}.xlsx".format(metrics_filename))
