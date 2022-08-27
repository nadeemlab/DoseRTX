# Read rt structs data from dicoms
import os
import SimpleITK as sitk
from pydicom import dcmread
import numpy as np
import cv2


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


def write_image(img_arr, out_dir, case, suffix, ref_ct):
	img_itk = sitk.GetImageFromArray(img_arr)
	img_itk.SetOrigin(ref_ct.GetOrigin())
	img_itk.SetSpacing(ref_ct.GetSpacing())
	img_itk.SetDirection(ref_ct.GetDirection())
	filename = os.path.join(out_dir, case + suffix)
	sitk.WriteImage(img_itk, filename)


in_dir = r'/nadeem_lab/Gourav/lungEcho2'
ct_in_dir = r'/nadeem_lab/Gourav/lung-manual-imrt-dicomextracted'
out_dir = r'/nadeem_lab/Gourav/lung-manual-imrt-dicomextracted'

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

	ds = get_rtstruct_dicom(in_dir, case)  # RT struct dicom dataset with all information
	if ds is None:
		print('\t RT struct not found.')
		continue
	ref_ct = get_ref_ct(ct_in_dir, case)	 # Ref CT to transform contour points

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

		#pts_img = np.zeros((oar_mask.shape[1], oar_mask.shape[2], 3), dtype=np.uint8)

		for i, elem in enumerate(anatomyStruc['ContourSequence']):  # Fill all planar contours for anatomy 'k' with corresponding label
			points = get_contour_points(elem)

			coords = transform_phy_pts_to_indexes(points, ref_ct)
			anatomy_mask = fill_planar_contour_as_mask(coords, anatomy_mask)
			# if coords[0][2] == 65 and k == 'Lung_R':
			# 	print(i, elem)
			# 	with open(str(i) + '_Lung_R_coords.txt', 'w') as f:
			# 		for row in coords:
			# 			print(row, file=f)
			# 			pts_img = cv2.circle(pts_img, center=(row[0], row[1]), radius=1,color=(0,0,255), thickness=-1)

		# if k == 'Lung_R':
		# 	cv2.imwrite('Lung_R.png', pts_img)

		if k == 'ptv':
			ptv_mask[np.where(anatomy_mask > 0)] = labels[k]
		else:
			oar_mask[np.where(anatomy_mask > 0)] = labels[k]

	write_image(oar_mask, out_dir, case, '_RTSTRUCTS.nrrd', ref_ct)
	write_image(ptv_mask, out_dir, case, '_PTV.nrrd', ref_ct)
