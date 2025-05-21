"""
Femur and tibia regions seperations from provided images

This script performs automatic segmentation of Femur and tibia region.
Includes denoising, thresholding, cleaning, seperation and saving .nii.gz file

Author: Prabesh Timilsina
Date: 2025-05-19
"""

import SimpleITK as sitk # library for reding medical images
import numpy as np # For mathematical and array operations
import os
from scipy.ndimage import gaussian_filter, binary_fill_holes, label #Smoothing, filling holes
from skimage.morphology import remove_small_objects # Removing noise
import argparse

"""
Setting threshold for bone detection
Remove noise
Setting sigma value for denoising
Setting saving directory
"""
THRESHOLD=290
SMALL_OBJ_SIZE= 900
GAUSSIAN_SIGMA= 1
SAVE_DIR= "./output"

def load_ct_scan(file_path):
    """
    Load CT scan .nii.gz file
    
    Args:
        file_path(str): Path to original CT scan file

    Returns:
        Img object and its array data.
    """
    img= sitk.ReadImage(file_path) # read input file
    array= sitk.GetArrayFromImage(img) # return pixel array
    return img, array

def denoise(array, sigma=1):
    """
    Gaussian filter for reducing noise

    Args:
        array: Input array
        sigma: sigma value for gaussian filter
    """
    return gaussian_filter(array, sigma= sigma)

def threshold_bone(array, threshold):
    """
    Intensity threshold to identify bone region

    Args:
        array: CT image array.
        threshold: HU threshold for segmentation

    Returns:
        np.array: Binary mask (0,1) for bone
    """
    return (array > threshold).astype(np.uint8)

def clean_mask(mask, min_size):
    """
    Cleaning binary mask slice by slice, filling holes and removing noise

    Args:
        mask: 3d binary mask
        min_size: Minimum object size that can be allowed

    Returns:
        np.array: Cleaned binary mask
    """
    cleaned = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        slice_mask= mask[i]
        slice_mask= binary_fill_holes(slice_mask)
        slice_mask= remove_small_objects(slice_mask.astype(bool), min_size=min_size)
        cleaned[i]= slice_mask.astype(np.uint8)
    return cleaned

def separate_bones(mask):
    """
    Seperate two largest bone segments

    Args:
        mask: Cleaned bone mask

    Returns:
        tuple: femur and tibia as binary array
    """
    labeled, num = label(mask) #find connected components in mask and assign new integer label to each component
    regions = []

    for i in range(1, num + 1): # looping through each components
        coords = np.argwhere(labeled == i) #gives z,y,and x corrdinate
        z_center = np.mean(coords[:, 0]) #extaract z component and store in dictionary
        z_min = np.min(coords[:, 0])
        z_max = np.max(coords[:, 0])
        volume = len(coords)

        regions.append({
            'label': i,
            'center_z': z_center,
            'z_min': z_min,
            'z_max': z_max,
            'volume': volume
        })

    regions = sorted(regions, key=lambda x: x['volume'], reverse=True)[:2] #sort region by volume and get 2 largest bones

    comp1= regions[0]
    comp2= regions[1]

    if comp1['center_z'] < comp2['center_z']: #lower center z is tibia and upper center z is femur
        tibia= (labeled == comp1['label']).astype(np.uint8)
        femur= (labeled == comp2['label']).astype(np.uint8)
    else:
        tibia= (labeled == comp2['label']).astype(np.uint8)
        femur= (labeled == comp1['label']).astype(np.uint8)

    return femur, tibia

def save_mask(mask, ref_img, filename, scale=False):
    """
    Save binary mask as .nii.gz file

    Args:
        mask: The binary mask
        ref_img: Originmal image
        filename: output file name
        scale: scaling decision (0 to 255)
    """
    out= mask.copy()
    if scale:
        out= (out * 255).astype(np.uint8) #converting binary back to uint8
    out_img = sitk.GetImageFromArray(out) # convert numpy array into SimpleITK image
    out_img.CopyInformation(ref_img) # copy metadata
    sitk.WriteImage(out_img, filename) #save to new file location

def segment_bones(input_path):
    """
    Pipeline to segment femur and tibia

    Args:
        input_path: Path to CT scan file
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    img, array= load_ct_scan(input_path)
    array_denoised= denoise(array, sigma=GAUSSIAN_SIGMA)
    bone_mask= threshold_bone(array_denoised, threshold=THRESHOLD)
    cleaned_mask= clean_mask(bone_mask, min_size=SMALL_OBJ_SIZE)
    femur_mask, tibia_mask= separate_bones(cleaned_mask)
    save_mask(femur_mask, img, os.path.join(SAVE_DIR, "femur_mask.nii.gz"), scale= True)
    save_mask(tibia_mask, img, os.path.join(SAVE_DIR, "tibia_mask.nii.gz"), scale= True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to CT scan")
    args= parser.parse_args()
    
    segment_bones(args.input)
