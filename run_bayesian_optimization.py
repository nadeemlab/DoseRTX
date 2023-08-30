import os
import numpy as np
import SimpleITK as sitk
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from datetime import datetime

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


def _create_p_bounds(optimisation_params):
    """
    Just reformat the optimisation variables into a format that the Bayesian optimiser wants
    """

    pbounds = {}
    for i, ParamName in enumerate(optimisation_params['ParameterNames']):
        pbounds[ParamName] = (optimisation_params['LowerBounds'][i], optimisation_params['UpperBounds'][i])

    return pbounds


def calculate_score(beam_0, beam_1, beam_2, beam_3, beam_4, beam_5, beam_6):
    
    planName = 'echoBeamMomLossW0_1'
    os.system('python3 ./preprocess/create_beamlet_matrix_bayes_rev3.py')
    os.system(
        'python3 test.py --dataroot ./nadeem_lab/Gourav/datasets/boo --netG unet_128 --name {} --phase test --mode eval --model doseprediction3d --input_nc 8 --output_nc 1 --direction AtoB --dataset_mode dosepred3d --norm batch'.format(
            planName))
    # os.system('python3 ./openkbp-stats/dvh-stats-open-kbp.py --planName {}'.format(planName))

    gt_dir = './nadeem_lab/Gourav/datasets/boo/test'
    pred_dir = './results/{}/test_latest/npz_images'.format(planName)
    cases = ['LUNG1-002']
    gt_oar_metrics = np.zeros((len(cases), 5 + 3),
                              dtype='float32')  # 5 OAR metrics (mean) and 3 PTV metrics (D1, D95, D99)
    # per patient
    pred_oar_metrics = np.zeros_like(gt_oar_metrics)

    gt_oar_metrics_max = np.zeros((len(cases), 5),
                                  dtype='float32')  # 5 OAR metrics (mean) and 3 PTV metrics (D1, D95, D99)
    # per patient
    pred_oar_metrics_max = np.zeros_like(gt_oar_metrics_max)

    ##D(0.1cc)
    mae_metrics = np.zeros((len(cases)), dtype='float32')
    mae_metrics_oar = np.zeros((len(cases), 5 + 3), dtype='float32')

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
    print(dvh_errors_all.mean())
    print(mae_metrics.mean())

    dvh_score = dvh_errors_all.mean()
    dose_score = mae_metrics.mean()
    return -(0.4 * dvh_score + 0.6 * dose_score)  # since we are maximizing


def function_to_be_optimized(beam_0, beam_1, beam_2, beam_3, beam_4, beam_5, beam_6):
    beam_0 = int(beam_0)  # convert fraction into int
    beam_1 = int(beam_1)
    beam_2 = int(beam_2)
    beam_3 = int(beam_3)
    beam_4 = int(beam_4)
    beam_5 = int(beam_5)
    beam_6 = int(beam_6)
    beams = [beam_0, beam_1, beam_2, beam_3, beam_4, beam_5, beam_6]

    check_duplicates = [True if beams.count(beam) > 1 else False for beam in beams]

    if check_duplicates.count(True) > 0:  # check if there is any True in the duplicates list
        return -10
    else:
        return calculate_score(beam_0, beam_1, beam_2, beam_3, beam_4, beam_5, beam_6)


# Start main script
start_time = datetime.now()

# set up optimisation params:
optimisation_params = {}
optimisation_params['ParameterNames'] = ['beam_0', 'beam_1', 'beam_2', 'beam_3', 'beam_4', 'beam_5', 'beam_6']
optimisation_params['UpperBounds'] = np.array([71, 71, 71, 71, 71, 71, 71])
optimisation_params['LowerBounds'] = np.array([0, 0, 0, 0, 0, 0, 0])
# do bayesian optimization based upon dvh and dose score
optimizer = BayesianOptimization(
    f=None,
    pbounds=_create_p_bounds(optimisation_params),
    verbose=2,
    random_state=1,
)

# utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
beam_filename = r'./nadeem_lab/Gourav/datasets/boo/beams.txt'
planName = 'echoBeamMomLossW0_1'
log_file = r'./paper_metrics/{}/log.txt'.format(planName)
best_so_far = -10
if os.path.exists(log_file):
    append_write = 'a'  # append if already exists
else:
    append_write = 'w'  # make a new file if not

with open(log_file, append_write) as f:
    print('\n' + '#'*70 + '\n', file=f)

for i in range(40):
    if i <= 15:
        xi = 0.1
    elif i <= 30:
        xi = 0.01
    else:
        xi = 0.001
    print('Iteration #: {}'.format(i))
    utility = UtilityFunction(kind="ei", xi=xi)
    # if i > 0:
    next_point = optimizer.suggest(utility)
    np.savetxt(beam_filename, np.asarray(list(next_point.values())), fmt='%i')
    target = function_to_be_optimized(**next_point)
    optimizer.register(params=next_point, target=target)
    # else:next_point = {'beam_0': 0, 'beam_1': 10, 'beam_2': 20, 'beam_3': 30, 'beam_4': 40}
    #     #     np.savetxt(beam_filename, np.asarray(list(next_point.values())), fmt='%i')
    #     #     target = function_to_be_optimized(**next_point)
    #     #     optimizer.register(params=next_point, target=target)
    #
    print(target, next_point)
    best_so_far = max(target, best_so_far)
    print('Best so far {}'.format(best_so_far))
    if abs(best_so_far - target) <= 0.01:  # check if for that iteration target is same as best so far then update optimal point
        best_so_far_param = next_point
    with open(log_file, 'a') as f:
        print('-' * 70, file=f)
        print('Target: {:<.2f}'.format(target), file=f)
        print('next point: {}'.format(next_point), file=f)
        print('best so far: {}'.format(best_so_far), file=f)
        print('best so far beams: {}'.format(best_so_far_param), file=f)

end_time = datetime.now()

print('Best solution using bayesian optimization {}'.format(optimizer.max))
print('Total time: {}'.format(end_time - start_time))
# for i, res in enumerate(optimizer.res):
#     print("Iteration {}: \n\t{}".format(i, res))
