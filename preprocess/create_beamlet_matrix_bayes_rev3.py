# Read the influence matrix from text files sent by Gourav
# Format is x y z value one per line
# x,y,z seem to be in physical units. So assuming we can use the echo dose files header info to map physical to
# grid indexes.
# Normalize the beamlet to have same amx min values as the corresponding echo dose

import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as spio
from scipy.sparse import csr_matrix
from numpy import loadtxt


def get_caselist(txt_file):
    datasets = []
    with open(txt_file, 'r') as f:
        for dset in f:
            datasets.append(dset.strip())
    return datasets


def resample(img, ref_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetReferenceImage(ref_image)
    img = resampler.Execute(img)

    return img


def get_dataset(in_dir, case, suffix, itk=False):
    filename = os.path.join(in_dir, case + suffix)
    img = sitk.ReadImage(filename)
    if itk is False:
        img = sitk.GetArrayFromImage(img)

    return img


def get_crop_settings(oar):
    # Use to get crop settings
    # Don't use cord or eso as they spread through more slices
    # If total number of slices is less than 128 then don't crop at all
    # Use start and end index from presence of any anatomy or ptv
    # If that totals more than 128 slices then leave as is.
    # If that totals less than 128 slices then add slices before and after to make total slices to 128

    oar1 = oar.copy()
    oar1[np.where(oar == 1)] = 0
    oar1[np.where(oar == 2)] = 0

    # For 2D cropping just do center cropping 256x256
    center = [0, oar.shape[1] // 2, oar1.shape[2] // 2]
    start = [0, center[1] - 150, center[2] - 150]
    end = [0, center[1] + 150, center[2] + 150]

    depth = oar1.shape[0]
    if depth < 128:
        start[0] = 0
        end[0] = depth

        return start, end

    first_slice = -1
    last_slice = -1
    for i in range(depth):
        frame = oar1[i]
        if np.any(frame):
            first_slice = i
            break
    for i in range(depth - 1, -1, -1):
        frame = oar1[i]
        if np.any(frame):
            last_slice = i
            break

    expanse = last_slice - first_slice + 1
    if expanse >= 128:
        start[0] = first_slice
        end[0] = last_slice

        return start, end

    # print('Get\'s here')
    slices_needed = 128 - expanse
    end_slices = slices_needed // 2
    beg_slices = slices_needed - end_slices

    room_available = depth - expanse
    end_room_available = depth - last_slice - 1
    beg_room_available = first_slice

    leftover_beg = beg_room_available - beg_slices
    if leftover_beg < 0:
        end_slices += np.abs(leftover_beg)
        first_slice = 0
    else:
        first_slice = first_slice - beg_slices

    leftover_end = end_room_available - end_slices
    if leftover_end < 0:
        first_slice -= np.abs(leftover_end)
        last_slice = depth - 1
    else:
        last_slice = last_slice + end_slices

    if first_slice < 0:
        first_slice = 0

    start[0] = first_slice
    end[0] = last_slice

    return start, end


def crop_resize_img(img, start, end, is_mask=False):
    # Crop to setting given by start/end coordinates list, assuming depth,height,width

    img_cropped = img[start[0]:end[0]+1, start[1]:end[1], start[2]:end[2]]
    img_cropped = np.moveaxis(img_cropped, 0, -1)  # Slices last

    order = 0
    if is_mask is False:
        order = 1
    img_resized = resize(img_cropped, (128,128,128), order=order, preserve_range=True, anti_aliasing=False).astype('float32')
    if is_mask is True:
        img_resized = img_resized.astype('uint8')

    img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

    return img_resized


def read_influence_matrix(in_dir, case):
    filename = os.path.join(in_dir, case, 'Dose.txt')  # Sparse
    # filename = os.path.join(in_dir, case, 'Dose1.txt')   # Dense Beam Echo
    # filename = os.path.join(in_dir, case, 'Dose2.txt')  # Dense beam Manual
    beamlet_info = np.genfromtxt(filename)

    return beamlet_info


in_dir = './nadeem_lab/Gourav/sample_nrrd_data'
# beamlet_info_in_dir = '/data/MSKCC-Intern-2021/Dose-Echo-Data/dose_beamlet_3D_martices/influenceMatrix_beamlet_info'
out_dir = './nadeem_lab/Gourav/datasets/boo/test'
# case_file = '../resources/train_echo_dose.txt'
case_file = '../resources/test_echo_dose.txt'
# cases = get_caselist(case_file)
cases = ['LUNG1-002']
normalize = True

filename = r'./nadeem_lab/Gourav/datasets/boo/beams.txt'

beams = loadtxt(filename, dtype=int)
beamMaps = loadtxt(os.path.join(in_dir, 'beams_ind_' + cases[0] + '.txt'), dtype=int)
points = loadtxt(os.path.join(in_dir, 'Points_' + cases[0] + '.txt'))
print(points.shape)
print('Beams in this iteration {}'.format(beams))
beams = list(beams)
infMatrix = pd.read_csv(os.path.join(in_dir, 'inf_matrix_' + cases[0]), sep='\t', header=None)
infMatrix = csr_matrix((infMatrix[2], (infMatrix[0] - int(1), infMatrix[1] - int(1))))
beamlets_to_consider = []
for b in range(len(beamMaps)):
    if b in beams:
        startB = beamMaps[b, 0]
        endB = beamMaps[b, 1]
        beamlets_to_consider.append(
            np.arange(startB, endB))

infMatrix = infMatrix[:, np.hstack(beamlets_to_consider)]
print(infMatrix.shape)
inf_sum = infMatrix.sum(axis=1)
beamlet_info = np.column_stack((points[:, 0:3], inf_sum.A1))

for idx, case in enumerate(cases):

    print('Processing case: {} {} of {} ...'.format(case, idx + 1, len(cases)))
    # dose = get_dataset(in_dir, case, '_dose_ECHO.nrrd', itk=True)
    dose = get_dataset(in_dir, case, '_dose_ECHO.nrrd', itk=True)  # echo dose
    # beamlet_info = read_influence_matrix(beamlet_info_in_dir, case)
    beamlet_arr = np.zeros(sitk.GetArrayFromImage(dose).shape, dtype=float)

    for row in beamlet_info:
        curr_pt = (row[0], row[1], row[2])
        curr_val = row[3]

        curr_indx = dose.TransformPhysicalPointToIndex(curr_pt)  # X,Y,Z

        beamlet_arr[curr_indx[2], curr_indx[1], curr_indx[0]] = curr_val

    if normalize is True:
        dose_arr = sitk.GetArrayFromImage(dose)
        tmaxm = dose_arr.max()
        tminm = dose_arr.min()

        rmin = beamlet_arr.min()
        rmax = beamlet_arr.max()
        rrange = rmax - rmin

        if rrange > 0:
            beamlet_arr = ((beamlet_arr - rmin) / rrange) * (
                    tmaxm - tminm) + tminm  # Beamlet array normalized to have same range as dose array
        else:
            print('\tBeamlet all zeros.')

    beamlet_itk = sitk.GetImageFromArray(beamlet_arr)
    beamlet_itk.SetOrigin(dose.GetOrigin())
    beamlet_itk.SetSpacing(dose.GetSpacing())
    beamlet_itk.SetDirection(dose.GetDirection())
    ct = get_dataset(in_dir, case, '_CT.nrrd', itk=True)
    dose_beamlet_resampled = resample(beamlet_itk, ct)
    beamlet_arr_resampled = sitk.GetArrayFromImage(dose_beamlet_resampled)
    print(beamlet_arr_resampled.shape)
    # added from other code
    ct = get_dataset(in_dir, case, '_CT.nrrd')
    oar = get_dataset(in_dir, case, '_RTSTRUCTS.nrrd')
    ptv = get_dataset(in_dir, case, '_PTV.nrrd')
    oar_copy = oar.copy()
    oar_copy[np.where(ptv == 1)] = 6

    start, end = get_crop_settings(oar_copy)

    ct = crop_resize_img(ct, start, end, is_mask=False)
    oar = crop_resize_img(oar, start, end, is_mask=True)
    ptv = crop_resize_img(ptv, start, end, is_mask=True)
    dose = crop_resize_img(dose, start, end, is_mask=False)
    beamlet = crop_resize_img(beamlet_arr_resampled, start, end, is_mask=False)
    beamlet[np.where(ptv == 1)] = 60  # PTV volume set to prescribed dose

    # modify npz file
    print('done')

    filename = os.path.join(out_dir, case + '.npz')
    npzfile = np.load(filename)
    mutable_file = dict(npzfile)
    if 'BEAM' in mutable_file:
        mutable_file['BEAM'] = beamlet
        new_file_name = os.path.join(out_dir, case)
        np.savez(filename, **mutable_file)
    else:
        array_dict = {'BEAM': beamlet}
        for k in npzfile:
            array_dict[k] = npzfile[k]

        np.savez(filename, **array_dict)
