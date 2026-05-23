# VesselWeaver
AbdVesselGen is an anatomy-guided abdominal vessel reconstruction framework for CT. It combines segmentation, anchor-point extraction, topology-aware pathfinding, and vessel graph generation to automatically trace major abdominal arteries and veins, with direct 3D Slicer SEG export support.

# AbdVesselGen: Anatomy-Guided Abdominal Vessel Reconstruction for CT

**AbdVesselGen** is an anatomy-guided abdominal vessel reconstruction framework for contrast-enhanced abdominal CT.

The pipeline combines anatomical segmentation, anchor-point extraction, rule-guided vessel pathfinding, topology-aware post-processing, and automated vessel naming to reconstruct major abdominal arteries and veins. Final outputs can be exported as 3D Slicer-compatible `.seg.nrrd` segmentations for visualization, inspection, and downstream analysis.

---

## Keywords

abdominal vessel reconstruction; abdominal CT angiography; vessel tracing; abdominal artery segmentation; portal vein segmentation; anatomy-guided pathfinding; topology-aware vessel reconstruction; 3D Slicer segmentation; TotalSegmentator; rule-based medical image processing; vascular graph reconstruction; abdominal vessel atlas.

---

## 📌 Overview

**AbdVesselGen** is designed to reconstruct and organize major abdominal vascular structures from CT using an interpretable, anatomy-guided workflow.

Unlike purely end-to-end deep learning segmentation pipelines, this project combines:

- TotalSegmentator-derived anatomical landmarks
- Vessel candidate segmentation
- Organ and hilum anchor-point extraction
- Anatomical plane generation
- HU-guided pathfinding
- Topology-aware vessel tracing
- Rule-based vessel naming
- 3D Slicer-compatible segmentation export

The goal is to create an inspectable abdominal vessel reconstruction pipeline that can help generate structured vascular maps from abdominal CT.

This project is designed for:

- Abdominal vessel tracing
- CT-based vascular atlas generation
- Radiology AI preprocessing
- Surgical anatomy visualization
- Rule-guided abdominal artery and vein reconstruction
- 3D Slicer-based review and correction workflows

---

## 🧠 Pipeline

```text
Abdominal CT (.nii / .nii.gz)
↓
GUI-based case loading
↓
TotalSegmentator anatomical segmentation
↓
Tissue / cavity / liver segment extraction
↓
Colon vessel / portal venous candidate segmentation
↓
Anchor-point and anatomical plane generation
↓
Vessel start-point detection
↓
HU-guided vessel pathfinding
↓
Arterial and venous branch reconstruction
↓
Component naming and post-processing
↓
Export:

Intermediate masks (.nii.gz)
Named trunk segmentation (.seg.nrrd)
Named abdominal vessel branch segmentation (.seg.nrrd)
```

## 🩻 Target Vessels and Structures

The current pipeline focuses on major abdominal vascular structures, including:
```text
Celiac trunk
SMA trunk
IMA trunk
Gastroduodenal artery
Hepatic artery branches
Splenic artery
Left renal vein
Right renal vein
Left renal artery
Right renal artery
Portal vein system
```
The framework is modular, so additional vessel branches can be added as independent scripts and inserted into the pipeline.


## 🧩 Core Concepts
# 1. Anatomy-guided reconstruction

The pipeline uses anatomical structures such as the aorta, IVC, liver segments, spleen, renal hilum, hepatic hilum, duodenum surface, and vertebral-level planes to define plausible vascular regions and endpoints.

# 2. Anchor-point based pathfinding

Instead of blindly tracing vessels from arbitrary seeds, the pipeline generates anatomical anchor points such as:
```text
hepatic hilum spheres
spleen hilum point
renal hilum centroid spheres
duodenum surface anchors
renal artery / vein start candidates
search-volume derived starting points
```

These anchors constrain vessel pathfinding to anatomically meaningful start and end locations.

# 3. HU-guided vessel tracing

Pathfinding is performed on CT intensity space. Low-HU or invalid regions are penalized or excluded, while enhancing vascular voxels are preferred.

Typical forbidden or penalized regions include:
```text
HU == -1024
outside ROI_CT
previously traced vessel paths
anatomically implausible regions
```

# 4. Independent segment layers

Final .seg.nrrd outputs store vessels as independent segment layers. This allows overlapping segments when anatomically or algorithmically necessary.

## 📁 Project Structure
```text
AbdVesselGen/
├── run.py
│
├── Segmentator/
│   ├── AllSeg.py
│   ├── MoveAllSeg.py
│   ├── CavitySeg.py
│   ├── MoveCavitySeg.py
│   ├── TissueSeg.py
│   ├── MoveTissueSeg.py
│   ├── LiverSegmentsSeg.py
│   ├── MoveLiverSegmentsSeg.py
│   └── ColonVesselSeg/
│       ├── scripts/
│       │   ├── inference.py
│       │   ├── utils_inference.py
│       │   ├── prune.py
│       │   └── skeletonize.py
│       └── weights/
│           └── veins_segmentation_model.pth
│
├── Preprocess/
│   ├── Crop_ROI_CT.py
│   ├── FindRenalHilum.py
│   ├── FindSpleenHilum.py
│   ├── SearchStartCombined.py
│   ├── SearchStart_RenalA_RenalV.py
│   ├── RenalStartExtract.py
│   ├── FindAnchorPointsFromSearchVolumes.py
│   ├── FindDynamicCenterDuodenumSurface.py
│   └── GenerateDuodenumSurfaceAnchors.py
│
├── VesselsFinder/
│   ├── FindWeb_GDA.py
│   ├── Pathfind_HepaticArtery.py
│   ├── RenalPathfind.py
│   └── Pathfind_SplenicArtery.py
│
├── Postprocess/
│   ├── TrunkSeperator.py
│   └── BranchesCombine.py
│
├── Seg_Done/
│   ├── <case_id>_All_segmentation/
│   ├── <case_id>_trunk_cavities/
│   ├── <case_id>_tissue_4_types/
│   ├── <case_id>_liver_segments/
│   └── <case_id>_veins_segmentation/
│
├── ROI_CT/
│   └── <case_id>_ROI_CT.nii.gz
│
├── AnchorPoints/
│   ├── <case_id>_seg32_highest_seg31_lowest_Zplanes.nii.gz
│   ├── <case_id>_seg29_highest_seg28_lowest_Zplanes.nii.gz
│   ├── <case_id>_search3231_volume.nii.gz
│   ├── <case_id>_search2928_volume.nii.gz
│   ├── <case_id>_anchor_points_from_search_volumes.txt
│   ├── <case_id>_renal_hilum_seg2_seg3_centroid_spheres_diameter6.nii.gz
│   ├── <case_id>_SpleenHilum.nii
│   └── <case_id>_hepatic_hilum_spheres.nii
│
└── Results/
    ├── Intermediate/
    │   ├── <case_id>_GDA.nii.gz
    │   ├── <case_id>_hepatic_vessels.nii.gz
    │   ├── <case_id>_renal_artery_vein_paths.nii.gz
    │   └── <case_id>_splenic_artery_path.nii.gz
    │
    ├── VesselsTrunk.seg.nrrd
    ├── AbdominalVessels.seg.nrrd
    ├── <case_id>_VesselsTrunk_summary.txt
    └── <case_id>_AbdominalVessels_summary.txt
```

## 📂 Input Format

The pipeline currently supports a single CT NIfTI file selected through the GUI.

Supported formats:
```text
.nii
.nii.gz
```

Example:
```text
D:\rsna_bowel_injury\nifti_dataset\43_24055.nii
```

When loaded through run.py, the case ID is inferred from the CT filename:
```text
43_24055.nii → case_id = 43_24055
```

All downstream outputs are automatically named using this case_id.
