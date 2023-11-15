import os
import SimpleITK as sitk
from pydicom import dcmread
import numpy as np
import cv2
import argparse

def read_dicom(in_dir, case):
	dicom_names = os.listdir(os.path.join(in_dir, case))
	dicom_paths = []
	for dcm in dicom_names:
		if dcm[:2] == 'CT':
			dicom_paths.append(os.path.join(in_dir, case, dcm))

	img_positions = []
	for dcm in dicom_paths:
		ds = dcmread(dcm)
		img_positions.append(ds.ImagePositionPatient[2])

	indexes = np.argsort(np.asarray(img_positions))
	dicom_names = list(np.asarray(dicom_paths)[indexes])

	reader = sitk.ImageSeriesReader()
	reader.SetFileNames(dicom_names)
	img = reader.Execute()

	return img


def read_dicom_dose(in_dir, case):
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
	return rt_dose_itk


def get_rtstruct_dicom(in_dir, case):

	rt_file = None
	files = os.listdir(os.path.join(in_dir, case))
	for file in files:
		if file[:2] == 'RS':
			rt_file = file

	if rt_file is None:
		return None

	rt_file = os.path.join(in_dir, case, rt_file)
	ds = dcmread(rt_file)

	return ds


def get_ref_ct(ct_in_dir, case):
	filename = os.path.join(ct_in_dir, case + '_CT.nrrd')
	ref_ct = sitk.ReadImage(filename)

	return ref_ct


def get_oar_roi_indexes(ds, oar_dict):
	for idx, struc in enumerate(ds.StructureSetROISequence):
		elem = struc['ROIName']
		if elem.value in oar_dict:
			oar_dict[elem.value] = idx

	return oar_dict

def get_oar_roi_indexes_modified(ds, oar_dict):
	new_oar_dict = {}

	# Basic case ROI name difference is only upper or lower case
	for k, v in oar_dict.items():
		# We are searching for variations of key 'k' in the rt structure dicom data
		for idx, struc in enumerate(ds.StructureSetROISequence):
			elem = struc['ROIName']
			roi_name = elem.value.lower()
			if k == roi_name:
				new_oar_dict[k] = idx
				oar_dict[k] = idx  # It means ROI index found for this anatomy
				break

	# Second basic case ROI name is for e.g. 'Cord1' or 'heart1' and similar variations with suffix '1'
	for k, v in oar_dict.items():
		# We are searching for variations of key 'k' in the rt structure dicom data
		for idx, struc in enumerate(ds.StructureSetROISequence):
			elem = struc['ROIName']
			roi_name = elem.value.lower()
			if roi_name == k+'1':
				new_oar_dict[k+'1'] = idx
				oar_dict[k] = idx  # It means ROI index found for this anatomy
				break

	# Third special case for Cord which can be sometimes named 'SpinalCord'
	v = oar_dict['cord']
	if v == -1:  # If still not found
		for idx, struc in enumerate(ds.StructureSetROISequence):
			elem = struc['ROIName']
			roi_name = elem.value.lower()
			if roi_name == 'spinalcord':
				new_oar_dict[roi_name] = idx
				oar_dict['cord'] = idx  # It means ROI index found for this anatomy

	# Remaining cases: find any ROI names that starts with target anatomy name for e.g. 'PTV_New' 'PTV_Primary' etc.
	# Will analyze these cases manually
	for k, v in oar_dict.items():
		# We are searching for variations of key 'k' in the rt structure dicom data
		if v == -1:  # If still not found
			for idx, struc in enumerate(ds.StructureSetROISequence):
				elem = struc['ROIName']
				roi_name = elem.value.lower()
				if roi_name != k+'1':
					if roi_name.startswith(k):
						new_oar_dict[roi_name] = idx

	# Based on the analysis of the previous case output, if the length of the new dictionary is 6 then all anatomies
	# were found just some names were different. In this case just transfer the roi index to oar_dict
	# If the lenght of new dictionary is greater than 6 then multiple PTVs were found. Analysis of the dicom data in
	# slicer3d shows that in such cases we can pick ROI name 'E_AllPTVs' which combines all PTV masks into one.
	# If length of new dictionary is less than 6 then that is one special case '38068983' which has lungs named as
	# left_lung and right_lung. Find those id's and use them in oar_dict.
	# These above 3 cases based on length of dictionary should give us all RT Structs for all cases.

	if len(new_oar_dict) == 6:
		for k, v in oar_dict.items():
			if v == -1:
				for new_k, new_v in new_oar_dict.items():
					# if new_k.startswith(k):
					# 	oar_dict[k] = new_v
					# 	break
					if k in new_k:
						oar_dict[k] = new_v
						break
	elif len(new_oar_dict) > 6:
		for idx, struc in enumerate(ds.StructureSetROISequence):
			elem = struc['ROIName']
			roi_name = elem.value.lower()
			if roi_name == 'e_allptvs':
				oar_dict['ptv'] = idx
			elif roi_name == 'ptv_total':
				oar_dict['ptv'] = idx
			elif roi_name == 'ptv_60' or 'ptv60':
				oar_dict['ptv'] = idx
			elif roi_name == 'ptvdvh':
				oar_dict['ptv'] = idx
			elif roi_name == 'ptv_dibh6000':
				oar_dict['ptv'] = idx
			elif roi_name == 'z_ptvoptr':
				oar_dict['ptv'] = idx
			elif roi_name == 'ptv final':
				oar_dict['ptv'] = idx

	else:
		for idx, struc in enumerate(ds.StructureSetROISequence):
			elem = struc['ROIName']
			roi_name = elem.value.lower()
			if roi_name == 'left_lung':
				oar_dict['lung_l'] = idx
			if roi_name == 'right_lung':
				oar_dict['lung_r'] = idx
			if roi_name == '1ptv_rt_lung6000':
				oar_dict['ptv'] = idx

	return new_oar_dict, oar_dict


def get_contour_points(ds_elem):
	# Get the closed contour points triplets from dataset elem for the given plane
	points = np.asarray(ds_elem.ContourData)  # Array as [x0, y0, z0, x1, y1, z1 ....] in physical units
	points = np.reshape(points, (-1, 3))  # Array of point triplets, [ [x0, y0, z0], [x1, y1, z1], ....]

	return points


def transform_phy_pts_to_indexes(points, ref_ct):
	# Transform physical points to pixel coordinates (array indexes) using the reference ct's physical params
	pixel_coords = np.zeros_like(points)
	for idx in range(points.shape[0]):
		coord = (points[idx][0], points[idx][1], points[idx][2])
		coord = ref_ct.TransformPhysicalPointToIndex(coord)
		pixel_coords[idx] = coord

	return pixel_coords.astype(np.int)


def fill_planar_contour_as_mask(pixel_coords, mask3d):

	arr = np.zeros((mask3d.shape[1], mask3d.shape[2]), dtype='int32')  # 2D shape
	poly = pixel_coords[:,:2]
	#print(poly.dtype, arr.dtype)
	img = cv2.fillPoly(img=arr, pts=np.int32([poly]), color=1)  # Polygon fill the planar contour#Gourav changed it np.int32 as arr and points had different type
	mask = img.astype(np.uint8)
	zcord = pixel_coords[0][2]
	mask3d[zcord] = mask3d[zcord] + mask  # Single slice can have multiple contours which are specified separately (PTV mostly)

	return mask3d

def resample(img, ref_image):
	resampler = sitk.ResampleImageFilter()
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetReferenceImage(ref_image)
	img = resampler.Execute(img)
	
	return img

def write_image(img_arr, out_dir, case, suffix, ref_ct):
	img_itk = sitk.GetImageFromArray(img_arr)
	img_itk.SetOrigin(ref_ct.GetOrigin())
	img_itk.SetSpacing(ref_ct.GetSpacing())
	img_itk.SetDirection(ref_ct.GetDirection())
	filename = os.path.join(out_dir, case + suffix)
	sitk.WriteImage(img_itk, filename)

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

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("--in_dir", required=True, help="Enter input dir having patient folders with their dicoms")
parser.add_argument("--out_dir", required=True, help="Enter out dir having patient folders with their dicoms")
args, _ = parser.parse_known_args()
in_dir = args.in_dir
out_dir = args.out_dir

if not os.path.exists(out_dir):
	os.makedirs(out_dir)

cases = os.listdir(in_dir)

labels = {
	'cord': 1,
	'esophagus': 2,
	'heart': 3,
	'lung_l': 4,
	'lung_r': 5,
	'ptv': 1
}  # PTV will be stored separately as its extent is not mutually exclusive with other anatomies

for idx, case in enumerate(cases):
    print('Processing case {}: {} of {} ...'.format(case, idx+1, len(cases)))
    ##read dicom CT and write it in out_dir
    img = read_dicom(in_dir, case)
    filename = os.path.join(out_dir, case + '_CT.nrrd')
    sitk.WriteImage(img, filename)
    
    
    ##read dicom dose and write in out dir
    doseImg = read_dicom_dose(in_dir, case)
    filename = os.path.join(out_dir, case + '_dose.nrrd')
    sitk.WriteImage(doseImg, filename)
    #sitk.WriteImage(dose_img, filename)
    ds = get_rtstruct_dicom(in_dir, case)  # RT struct dicom dataset with all information
    if ds is None:
       print('\t RT struct not found.')
       continue
    ref_ct = get_ref_ct(out_dir, case)	 # Ref CT to transform contour points
    #oars = ['Cord', 'Esophagus', 'Heart', 'Lung_L', 'Lung_R', 'PTV']
    oars = ['cord', 'esophagus', 'heart', 'lung_l', 'lung_r', 'ptv']
    target_oars = dict.fromkeys(oars, -1)  # Will store index of target OAR contours from dicom dataset
    new_target_oars, target_oars = get_oar_roi_indexes_modified(ds, target_oars)
    # Check if all target oars found (several cases have renamed the structures e.g. case '38097625'
    all_found = True
    for k, v in target_oars.items():
        if v == -1:
           all_found = False
        if all_found is False:
           print('Case: ', case, ' One or more RT struct not found.')
           continue
        oar_mask = np.zeros(sitk.GetArrayFromImage(ref_ct).shape, np.uint8)
        ptv_mask = np.zeros_like(oar_mask)
        for k, v in target_oars.items():
                anatomyStruc = ds['ROIContourSequence'][v]
                anatomy_mask = np.zeros_like(oar_mask)
        for i, elem in enumerate(anatomyStruc['ContourSequence']):  # Fill all planar contours for anatomy 'k' with corresponding label
                points = get_contour_points(elem)
                coords = transform_phy_pts_to_indexes(points, ref_ct)
                anatomy_mask = fill_planar_contour_as_mask(coords, anatomy_mask)
                if k == 'ptv':
                        ptv_mask[np.where(anatomy_mask > 0)] = labels[k]
                else:
                        oar_mask[np.where(anatomy_mask > 0)] = labels[k]
        write_image(oar_mask, out_dir, case, '_RTSTRUCTS.nrrd', ref_ct)
        write_image(ptv_mask, out_dir, case, '_PTV.nrrd', ref_ct)
    
    dose_resampled = resample(doseImg, ref_ct)
    filename = os.path.join(in_dir, case + '_dose_resampled.nrrd')
    sitk.WriteImage(dose_resampled, filename)
    ##process all the nrrd files
    try:
            #print('Processing case {}: {} of {} ...'.format(case, idx+1, len(cases)))
            process_case(in_dir, out_dir, case)
    except:
            print('Processing of case {} failed'.format(case))
            pass
