import argparse
import SimpleITK as sitk
import os

def expand_mask_physically(input_mask_path, output_path=None, expansion_mm=2.0):
    """
    Expands a binary mask by a specified physical distance (in mm) using a distance map.
    """
    mask= sitk.ReadImage(input_mask_path, sitk.sitkUInt8)

    if sitk.GetArrayFromImage(mask).max() > 1:
        mask= sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255, 
                                insideValue=1, outsideValue=0)


    distance_map= sitk.SignedMaurerDistanceMap(
        mask,
        useImageSpacing=True,
        squaredDistance=False,
        insideIsPositive=False
    )

    expanded= sitk.BinaryThreshold(
        distance_map,
        lowerThreshold=-1e6,
        upperThreshold=expansion_mm,
        insideValue=1,
        outsideValue=0
    )
    
    expanded= sitk.Cast(expanded, sitk.sitkUInt8)
    expanded= expanded*255
    expanded.CopyInformation(mask)

    if output_path:
        sitk.WriteImage(expanded, output_path)
        print(f"Expanded mask saved: {output_path}")
    
    return expanded



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", nargs='+', required=False, default=["output/femur_mask.nii.gz", "output/tibia_mask.nii.gz"], help="Paths to input masks")
    parser.add_argument("--expansion", type=float, default=2.0, help="Expansion distance in mm eg:(2.0)")
    args = parser.parse_args()

    masks = ["femur_mask.nii.gz", "tibia_mask.nii.gz"]
    for mask in masks:
        filename= os.path.basename(mask)
        output_path= os.path.join("output", f"expanded_2mm_{filename}")
        expand_mask_physically(mask, output_path)
    
   
