"""
Randomly expand binary mask be a distance within limit

This script reads binary mask, computes physical distance map
augment random number and use randomness for expansion

Author: Prabesh Timilsina
Date: 2025-05-20
"""

import argparse
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np


def randomized_contour_adjustment(original_mask_path, output_path, max_expansion_mm=2.0, random_expansion_mm=1.5):
    """
    Randomly expand binary mask within a maximum limit

    Args:
        original_mask_path: Path to the original binary mask
        output_path: File path to save
        max_expansion_mm : Upper limit for physical expansion
        random_expansion_mm: Maximum random expansion
    
    Return:
        randomized_mask: Final mask after random expansion
    """
    
    #load original mask
    original= sitk.ReadImage(original_mask_path, sitk.sitkUInt8)
    original= sitk.BinaryThreshold(original, lowerThreshold=1, upperThreshold = 255, insideValue = 1, outsideValue=0)#ensure the mask is binary

    #compute distance map where inside is negative and outside is positive.
    distance_map= sitk.SignedMaurerDistanceMap(
        original, useImageSpacing=True, squaredDistance=False, insideIsPositive=False
    )

    #Maximum expanded mask using full expansion distance
    expanded_mask= sitk.BinaryThreshold(
        distance_map, lowerThreshold=-1e6, upperThreshold= max_expansion_mm,
        insideValue=1, outsideValue=0
    )

    #Sample random threshold distance
    threshold= np.random.uniform(1, random_expansion_mm)

    #create random mask with generated random threshold
    randomized_mask= sitk.BinaryThreshold(
        distance_map, lowerThreshold=-1e6, upperThreshold=threshold,
        insideValue=1, outsideValue=0
    )

    #Making sure random max doesn't extend beyond maximum
    randomized_mask= sitk.And(randomized_mask, expanded_mask)

    #Convert back to 8 bit for ImageJ
    randomized_mask= sitk.Cast(randomized_mask *255, sitk.sitkUInt8)
    randomized_mask.CopyInformation(original)
    sitk.WriteImage(randomized_mask, output_path)
    print(f"Saved randomized mask: {output_path}")

    return randomized_mask

if __name__ == "__main__":
    #passing arguments in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_expansion", type=float, default=2.0, help="Maximum expansion distance in mm")
    parser.add_argument("--random_expansion", type=float, default=1.5, help="Maximum random expansion value in mm")
    args = parser.parse_args()

    masks= [
        "output/femur_mask.nii.gz",
        "output/tibia_mask.nii.gz"
    ]

    for mask_path in masks:
        output_path = os.path.join("output", f"randomized_2_{os.path.basename(mask_path)}")
        randomized_contour_adjustment(
            original_mask_path= mask_path,
            output_path= output_path,
            max_expansion_mm= args.max_expansion,
            random_expansion_mm= args.random_expansion
        )
