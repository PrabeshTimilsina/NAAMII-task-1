"""
Outward expansion of mask by a fixed parameter

This script takes binary bone mask and expands it outwards by certain distance

Author: Prabesh Timilsina
Date: 2025-05-20
"""

import argparse
import SimpleITK as sitk
import os

def expand_mask_physically(input_mask_path, output_path=None, expansion_mm=4.0):
    """
    Takes the mask file, expands it by certain physical distance
    Args:
        input_mask_path: path to input mask
        output_mask: path to saving result
        expansion_mm: unit to perform expansion by
    Returns:
        expanded: expanded mask
    """
    mask= sitk.ReadImage(input_mask_path, sitk.sitkUInt8) #loading mask as 8-bit specifically for visulizing in ImageJ

    #Making sure mask is binary by converting random values to 0-1
    if sitk.GetArrayFromImage(mask).max() > 1: 
        mask= sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255, 
                                insideValue=1, outsideValue=0)


    #Image spacing by computing distance of each background voxel to nearest surface
    distance_map= sitk.SignedMaurerDistanceMap(
        mask,
        useImageSpacing=True,
        squaredDistance=False,
        insideIsPositive=False
    )

    #Select voxel within given limit
    expanded= sitk.BinaryThreshold(
        distance_map,
        lowerThreshold=-1e6,
        upperThreshold=expansion_mm,
        insideValue=1,
        outsideValue=0
    )
    
    #Converting back to 8 bit
    expanded= sitk.Cast(expanded, sitk.sitkUInt8)
    expanded= expanded*255
    expanded.CopyInformation(mask)# copying metadata for proper physical alignment

    #Save expanded mask
    if output_path:
        sitk.WriteImage(expanded, output_path)
        print(f"Expanded mask saved: {output_path}")
    
    return expanded



if __name__ == "__main__":
    #Passing as argument in commandline
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", nargs='+', required=False, default=["output/femur_mask.nii.gz", "output/tibia_mask.nii.gz"], help="Paths to input masks")
    parser.add_argument("--expansion", type=float, default=4.0, help="Expansion distance in mm eg:(2.0)")
    args = parser.parse_args()

    masks = ["femur_mask.nii.gz", "tibia_mask.nii.gz"]
    for mask in masks:
        filename= os.path.basename(mask)
        output_path= os.path.join("output", f"expanded_4mm_{filename}")
        expand_mask_physically(mask, output_path)
    
   
