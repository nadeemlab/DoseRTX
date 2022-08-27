# Prepapre 3D data for training crop and resize to 128x128x128. Also create ground truth dvh data and save ground
# truth CT/Dose/OAR/DVH

import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import torch
import torch.nn as nn


def get_dataset(in_dir, case, suffix):
	filename = os.path.join(in_dir, case + suffix)
	img = None
	if os.path.exists(filename):
		img = sitk.ReadImage(filename)
		img = sitk.GetArrayFromImage(img)

	return img


def get_caselist(txt_file):
	datasets = []
	with open(txt_file, 'r') as f:
		for dset in f:
			datasets.append(dset.strip())
	return datasets


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
	for i in range(depth-1, -1, -1):
		frame = oar1[i]
		if np.any(frame):
			last_slice = i
			break

	expanse = last_slice - first_slice + 1
	if expanse >= 128:
		start[0] = first_slice
		end[0] = last_slice

		return start, end

	#print('Get\'s here')
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
	img_resized = resize(img_cropped, (128,128,128), order=order, preserve_range=True, anti_aliasing=False).astype(np.float32)
	img_resized = resize(img_cropped, (128, 128, 128), order=order, preserve_range=True, anti_aliasing=False).astype(
		np.float32)
	if is_mask is True:
		img_resized = img_resized.astype(np.uint8)

	img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

	return img_resized


def get_torch_tensor(npy_tensor, device):
	out = torch.from_numpy(npy_tensor)
	out.to(device)

	return out
def get_dvh(dose, oar, ptv):
	# Compute and return the dvh for all 6 OAR structures
	device = torch.device('cuda:0')
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
		hist[i] = (num / vols)

	hist_numpy = hist.cpu().numpy()
	bins_np = bins.cpu().numpy()

	return hist_numpy, bins_np
# def get_dvh(dose, oar):
# 	# Compute and return the dvh for all 6 OAR structures
# 	device = torch.device('cuda:0')
# 	dose = get_torch_tensor(dose, device)
# 	oar = get_torch_tensor(oar, device).long()
# 	oar = torch.nn.functional.one_hot(oar, 7)[..., 1:]  # Remove BG
# 	oar = oar.permute(3, 0, 1, 2).to(torch.float)
#
# 	vols = torch.sum(oar, axis=(1, 2, 3))
# 	n_bins = 351
# 	hist = torch.zeros((n_bins, 6)).to(device)
# 	bins = torch.linspace(0, 70, n_bins)
# 	bin_w = bins[1] - bins[0]
#
# 	for i in range(bins.shape[0]):
# 		diff = torch.sigmoid((dose - bins[i]) / bin_w)
# 		diff = torch.cat(6 * [diff.unsqueeze(axis=0)]) * oar
# 		num = torch.sum(diff, axis=(1, 2, 3))
# 		hist[i] = (num / vols)
#
# 	hist_numpy = hist.cpu().numpy()
# 	bins_np = bins.cpu().numpy()
#
# 	return hist_numpy, bins_np


# def process_case(in_dir, out_dir, case):
# 	ct = get_dataset(in_dir, case, '_CT.nrrd')
# 	dose = get_dataset(in_dir, case, '_dose_resampled.nrrd')
# 	# dose = get_dataset(in_dir, case, '_echo_dose_resampled.nrrd')
# 	oar = get_dataset(in_dir, case, '_RTSTRUCTS.nrrd')
# 	ptv = get_dataset(in_dir, case, '_PTV.nrrd')
#
# 	oar[np.where(ptv == 1)] = 6
#
# 	start, end = get_crop_settings(oar)
#
# 	ct = crop_resize_img(ct, start, end, is_mask=False)
# 	oar = crop_resize_img(oar, start, end, is_mask=True)
# 	dose = crop_resize_img(dose, start, end, is_mask=False)
#
# 	ct = np.clip(ct, a_min=-1000, a_max=3071)
# 	ct = (ct + 1000) / 4071
# 	ct = ct.astype(np.float32)
#
# 	dose = np.clip(dose, a_min=0, a_max=70)
#
# 	hist, bins = get_dvh(dose, oar)
#
# 	filename = os.path.join(out_dir, case)
# 	np.savez(filename, CT=ct, DOSE=dose, OAR=oar, HIST=hist, BINS=bins)

def process_case(in_dir, out_dir, case):
	ct = get_dataset(in_dir, case, '_CT.nrrd')
	dose = get_dataset(in_dir, case, '_dose_resampled.nrrd')
	# dose = get_dataset(in_dir, case, '_dose_resampled.nrrd')  # Manual dose
	oar = get_dataset(in_dir, case, '_RTSTRUCTS.nrrd')
	ptv = get_dataset(in_dir, case, '_PTV.nrrd')
	# beamlet = get_dataset(in_dir, case, '_echo_dose_beamlet_resampled.nrrd')
	# beamlet = get_dataset(in_dir, case, '_manual_dose_beamlet_resampled.nrrd')
	# beamlet = get_dataset(in_dir, case, '_echo_dose_beamlet_sparse_resampled.nrrd')
	# beamlet = get_dataset(in_dir, case, '_manual_dose_beamlet_sparse_resampled.nrrd')

	oar_copy = oar.copy()
	oar_copy[np.where(ptv == 1)] = 6

	start, end = get_crop_settings(oar_copy)

	ct = crop_resize_img(ct, start, end, is_mask=False)
	oar = crop_resize_img(oar, start, end, is_mask=True)
	ptv = crop_resize_img(ptv, start, end, is_mask=True)
	dose = crop_resize_img(dose, start, end, is_mask=False)
	# beamlet = crop_resize_img(beamlet, start, end, is_mask=False)
	# beamlet[np.where(ptv == 1)] = 60  # PTV volume set to prescribed dose

	# Scale PTV volume (region) in dose to have average prescibed 60 Gy

	num_ptv = np.sum(ptv)
	dose_copy = dose.copy()
	dose_copy *= ptv
	sum = np.sum(dose_copy)
	scale_factor = (60 * num_ptv) / sum

	dose_copy *= scale_factor

	dose[np.where(ptv == 1)] = dose_copy[np.where(ptv == 1)]

	ct = np.clip(ct, a_min=-1000, a_max=3071)
	ct = (ct + 1000) / 4071
	ct = ct.astype(np.float32)

	dose = np.clip(dose, a_min=0, a_max=70)

	hist, bins = get_dvh(dose, oar, ptv)

	filename = os.path.join(out_dir, case)
	np.savez(filename, CT=ct, DOSE=dose, OAR=oar, PTV=ptv, HIST=hist, BINS=bins)

























