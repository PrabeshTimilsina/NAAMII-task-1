import argparse
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np

def verify_expansion(original_path, random_path, expanded_path):
    """Verify random mask lies between original and expanded"""
    original= sitk.GetArrayFromImage(sitk.ReadImage(original_path))
    random= sitk.GetArrayFromImage(sitk.ReadImage(random_path))
    expanded= sitk.GetArrayFromImage(sitk.ReadImage(expanded_path))
    
    # Check voxel counts
    print(f"\nVerification for {os.path.basename(random_path)}:")
    print(f"Original voxels: {np.sum(original)}")
    print(f"Random voxels:   {np.sum(random)}")
    print(f"Expanded voxels: {np.sum(expanded)}")
    
    # Check containment
    assert np.all(random>= original), "Random mask shrinks below original!"
    assert np.all(random<= expanded), "Random mask exceeds expansion limit!"
    
    # Visual check (middle slice)
    slice_idx= original.shape[0] // 2
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.imshow(original[slice_idx], cmap='gray')
    plt.title('Original Mask')
    
    plt.subplot(1,3,2)
    plt.imshow(random[slice_idx], cmap='gray')
    plt.title('Randomized Mask')
    
    plt.subplot(1,3,3)
    plt.imshow(expanded[slice_idx], cmap='gray')
    plt.title('Expanded Mask')
    
    plt.savefig(f"verification_{os.path.basename(random_path)}.png")
    plt.close()

def randomized_contour_adjustment(original_mask_path, output_path, max_expansion_mm=2.0, random_expansion_mm=1.0):
    """
    Generates a randomized mask lying between the original and max-expanded (e.g., 2mm) mask.
    The amount of random expansion is bounded and both values are parameters.
    """
    
    original= sitk.ReadImage(original_mask_path, sitk.sitkUInt8)
    original= sitk.BinaryThreshold(original, lowerThreshold=1, upperThreshold = 255, insideValue = 1, outsideValue=0)

    distance_map= sitk.SignedMaurerDistanceMap(
        original, useImageSpacing=True, squaredDistance=False, insideIsPositive=False
    )

    expanded_mask= sitk.BinaryThreshold(
        distance_map, lowerThreshold=-1e6, upperThreshold= max_expansion_mm,
        insideValue=1, outsideValue=0
    )

    threshold= np.random.uniform(0, random_expansion_mm)

    randomized_mask= sitk.BinaryThreshold(
        distance_map, lowerThreshold=-1e6, upperThreshold=threshold,
        insideValue=1, outsideValue=0
    )

    randomized_mask= sitk.And(randomized_mask, expanded_mask)

    randomized_mask= sitk.Cast(randomized_mask *255, sitk.sitkUInt8)
    randomized_mask.CopyInformation(original)
    sitk.WriteImage(randomized_mask, output_path)
    print(f"Saved randomized mask: {output_path}")

    verify_expansion(
        original_mask_path,
        output_path,
        "output/expanded_2mm_femur_mask.nii.gz" if "femur" in output_path 
        else "output/expanded_2mm_tibia_mask.nii.gz"
    )
    return randomized_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_expansion", type=float, default=2.0, help="Maximum expansion distance in mm")
    parser.add_argument("--random_expansion", type=float, default=1.0, help="Maximum random expansion value in mm")
    args = parser.parse_args()

    masks= [
        "output/femur_mask.nii.gz",
        "output/tibia_mask.nii.gz"
    ]

    for mask_path in masks:
        output_path = os.path.join("output", f"randomized_{os.path.basename(mask_path)}")
        randomized_contour_adjustment(
            original_mask_path= mask_path,
            output_path= output_path,
            max_expansion_mm= args.max_expansion,
            random_expansion_mm= args.random_expansion
        )
