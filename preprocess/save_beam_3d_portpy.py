import portpy_photon as pp
import numpy as np
import os
pp.__version__

# specify the patient data location
# (you first need to download the patient database from the link provided in the GitHub page)
data_dir = r'../data'
patient_id='Lung_Patient_1'
my_plan = pp.Plan(patient_id, data_dir=data_dir)

# get dose in 3d
dose_1d = my_plan.inf_matrix.A*np.ones((my_plan.inf_matrix.A.shape[1]))
dose_1d = dose_1d.astype('float16')
dose_3d = my_plan.inf_matrix.dose_1d_to_3d(dose_1d = dose_1d)
dose_3d = dose_3d.astype('float16')

in_dir = '/nadeem_lab/Gourav/sample_nrrd_data'

for i in range(0,72):
    print('creating 3d beam # {}'.format(i))
    my_plan = pp.Plan(patient_id, data_dir=data_dir, beam_ids=[i])
    dose_1d = my_plan.inf_matrix.A*np.ones((my_plan.inf_matrix.A.shape[1]))
    dose_1d = dose_1d.astype('float16')
    dose_3d = my_plan.inf_matrix.dose_1d_to_3d(dose_1d = dose_1d)
    dose_3d = dose_3d.astype('float16')
    filename = 'beam_{}.npy'.format(i)
    np.save(os.path.join(in_dir, 'beams_3d', filename),dose_3d)
    