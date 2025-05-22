"""
Finding medial and lateral lowest points on the tibial surface.

This script finds the Medial and lateral lowest point of different varients of tibia mask 
generated from the original mask given for assignment

Author: Prabesh Timilsina
Date: 2025-05-22
"""

import csv
import SimpleITK as sitk
import numpy as np
import os

def find_medial_lateral_lowest_points(mask_path):
    """
    Finds medial and lateral lowest points on the tibial surface.

    Args:
        mask_path: Path to the binary tibia mask (.nii.gz)

    Returns:
        tuple: (medial_point_mm, lateral_point_mm) as numpy arrays in physical coordinates (mm)
    """
    # Load the mask image and ensuring that the value is binary for proper calculation
    # Also convert to 3d numpy array
    mask_img = sitk.ReadImage(mask_path) 
    mask_img = sitk.BinaryThreshold(mask_img, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    mask_array = sitk.GetArrayFromImage(mask_img).astype(bool)

    # Getting voxel indexes from metadata nd converting to millimeters
    origin = np.array(mask_img.GetOrigin())         
    spacing = np.array(mask_img.GetSpacing())      
    direction = np.array(mask_img.GetDirection()).reshape(3, 3)

    #remove voxel from inside because we want landmark of lowest point on surface not inside
    eroded_img = sitk.BinaryErode(mask_img, [1, 1, 1])
    eroded_array = sitk.GetArrayFromImage(eroded_img).astype(bool) #subtract eroded version from original
    surface = mask_array & (~eroded_array) #mask of outer most voxel
    
    surface_indices = np.argwhere(surface) #get coordinates of all surface voxel

    if surface_indices.size == 0:
        raise ValueError(f"No surface voxels found in mask: {mask_path}")

    # Use x coordinates to seperate between medial and lateral side that usually gives left and right direction
    x_coords = surface_indices[:, 2]
    median_x = np.median(x_coords) 
    medial_side = surface_indices[x_coords <= median_x]
    lateral_side = surface_indices[x_coords > median_x]

    #Find lowest point in each side medial and lateral
    medial_lowest_idx = medial_side[np.argmin(medial_side[:, 0])]
    lateral_lowest_idx = lateral_side[np.argmin(lateral_side[:, 0])]

    #converting to mm
    def voxel_to_physical(voxel_idx):
        """
        Converts voxel indices to physical coordinates

        Args:
            voxel_idx: voxel indices

        Return:
            Physical points for given voxel
        """
        physical_point = origin + direction.dot(voxel_idx[::-1] * spacing) #Revercing because Numpy is z,y,x(in reversed order)
        return physical_point

    medial_phys = voxel_to_physical(medial_lowest_idx)
    lateral_phys = voxel_to_physical(lateral_lowest_idx)

    return medial_phys, lateral_phys

def main():
    mask_files = [
        "output/tibia_mask.nii.gz",
        "output/expanded_2mm_tibia_mask.nii.gz",
        "output/expanded_4mm_tibia_mask.nii.gz",
        "output/randomized_tibia_mask.nii.gz",
        "output/randomized_2_tibia_mask.nii.gz"
    ]

    results= []

    for mask_file in mask_files:
        if not os.path.exists(mask_file):
            print(f"File not found: {mask_file}")
            continue

        medial, lateral = find_medial_lateral_lowest_points(mask_file)
        print(f"{os.path.basename(mask_file)}:")
        print(f" Medial lowest point (x,y,z): {medial}")
        print(f" Lateral lowest point (x,y,z): {lateral}\n")

        results.append([
            os.path.basename(mask_file),
            *medial,
            *lateral
        ])

    #Saving a CSV file for better readibility later
    with open("output/tibia_landmarks.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Mask File",
            "Medial X", "Medial Y", "Medial Z",
            "Lateral X", "Lateral Y", "Lateral Z"
        ])
        writer.writerows(results)

if __name__ == "__main__":
    main()
