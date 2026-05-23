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

## 🚀 Usage
# 1. Install dependencies

Recommended environment:
```text
conda create -n abdvesselgen python=3.10
conda activate abdvesselgen
```

Install core dependencies:

```text
pip install numpy scipy nibabel pynrrd nilearn tqdm
```

Install medical segmentation dependencies as needed:

```text
pip install torch monai simpleitk
pip install TotalSegmentator
```

If using the colon vessel segmentation module, place the pretrained model under:
```text
Segmentator/ColonVesselSeg/weights/veins_segmentation_model.pth
```
# 2. Run the GUI pipeline

```text
python run.py
```

Then:

1. Select the abdominal CT NIfTI file
2. Confirm the selected case
3. Click Run Pipeline
4. Monitor real-time logs in the GUI
5. Review outputs in Results/

The GUI supports automatic step skipping if required outputs already exist.

## 📤 Output
# 1. ROI CT
```text
ROI_CT/<case_id>_ROI_CT.nii.gz
```
This file contains the cropped / masked CT region used for downstream vessel pathfinding.

# 2. Anchor points
```text
AnchorPoints/
```

Contains anatomical anchors and candidate search masks, such as:

```text
<case_id>_search3231_volume.nii.gz
<case_id>_search2928_volume.nii.gz
<case_id>_anchor_points_from_search_volumes.txt
<case_id>_renal_hilum_seg2_seg3_centroid_spheres_diameter6.nii.gz
<case_id>_SpleenHilum.nii
<case_id>_hepatic_hilum_spheres.nii
```

# 3. Intermediate vessel masks
Results/Intermediate/

Typical outputs include:
```text
<case_id>_GDA.nii.gz
<case_id>_hepatic_vessels.nii.gz
<case_id>_renal_artery_vein_paths.nii.gz
<case_id>_splenic_artery_path.nii.gz
```
# 4. Vessel trunk segmentation
```text
Results/VesselsTrunk.seg.nrrd
```
This file contains named arterial trunk segments.

Naming rule:
```text
If 3 components:
highest Z-axis component  = Celiac_trunk
middle Z-axis component   = SMA_trunk
lowest Z-axis component   = IMA_trunk

If 2 components:
highest Z-axis component  = Celiac+SMA_trunk
lowest Z-axis component   = IMA_trunk
```

# 5. Final abdominal vessel segmentation
```text
Results/AbdominalVessels.seg.nrrd
```
This file combines major vessel branches into a 3D Slicer-compatible segmentation.

Segment naming rules:
```text
GDA.nii.gz:
GDA_1, GDA_2, GDA_3, ...

hepatic_vessels.nii.gz:
Hepatic_artery_1, Hepatic_artery_2, ...

renal_artery_vein_paths.nii.gz:
Segment_1 → Lt_Renal_vein
Segment_2 → Rt_Renal_vein
Segment_3 → Lt_Renal_artery
Segment_4 → Rt_Renal_artery

splenic_artery_path.nii.gz:
Splenic_artery

portal venous segmentation:
Portal_Vein_System
```

Because each vessel is stored as an independent segment layer, overlapping segments are allowed.

## 🧠 Method Highlights
# 1. TotalSegmentator-based anatomical segmentation

The pipeline uses TotalSegmentator-derived anatomical labels as spatial priors.

Key structures may include:
```text
aorta
inferior vena cava
liver
spleen
kidneys
vertebrae
trunk cavities
tissue classes
abdominal organs
```

These structures are used for ROI generation, anchor extraction, anatomical plane construction, and pathfinding constraints.

# 2. ROI CT generation

The pipeline creates an ROI CT by combining anatomical masks and excluding irrelevant or invalid regions.

The ROI CT is used as the main search space for pathfinding.

Invalid regions are typically encoded as:
```text
HU = -1024
```

Pathfinding scripts avoid these regions.

# 3. Anatomical plane generation

Several scripts generate anatomical Z-plane constraints based on TotalSegmentator labels.

Examples:
```text
seg32_highest_seg31_lowest_Zplanes
seg29_highest_seg28_lowest_Zplanes
```
These planes restrict where certain start-search algorithms are allowed to operate.

# 4. Vessel start-point detection

The pipeline searches for plausible vessel origins near major vascular structures.

Examples:
```text
renal artery / vein candidate boxes
search3231_volume
search2928_volume
anchor_points_from_search_volumes.txt
```

Start points are selected using connected component centroids and validated against ROI CT intensity values.

# 5. Hilum and organ anchor extraction

The pipeline automatically extracts anatomical target points such as:

```text
renal hilum centroid spheres
spleen hilum point
hepatic hilum spheres
duodenal internal surface anchors
```

These anchor points serve as vessel endpoints or intermediate constraints.

# 6. HU-guided pathfinding

Vessel branches are reconstructed using pathfinding over CT intensity space.

The pathfinding cost function generally:

```text
penalizes low-HU regions
avoids HU == -1024
prefers enhancing vascular voxels
avoids previously traced vessel masks when needed
```

This allows the pipeline to trace plausible vessel routes between anatomical start and endpoint anchors.

# 7. Direction-constrained splenic artery tracing

For splenic artery pathfinding, an initial forward-only constraint can be used.

Example:
```text
first 10 steps must move toward decreasing Y-axis value
```
This helps prevent the path from immediately traveling backward into incorrect high-HU regions.

# 8. Overlap-preserving Slicer export

Final .seg.nrrd outputs are written using independent segment layers:
```text
(num_segments, X, Y, Z)
```
This preserves overlapping vessels and avoids accidental overwriting that can occur in single-label 3D labelmaps.

## 🖥️ GUI Features

The included Tkinter GUI supports:
```text
manual CT NIfTI selection
automatic case ID extraction
one-click pipeline execution
real-time log display
progress-bar cleanup for subprocess logs
automatic output directory refresh
step skipping if output files already exist
total runtime measurement
```
This makes the pipeline easier to run repeatedly during development and debugging.

## ⚙️ Processing Time

Processing time depends on:
```text
CT volume size
TotalSegmentator runtime
GPU availability
number of vessel pathfinding steps
whether intermediate outputs already exist
```
The pipeline prioritizes anatomical interpretability and reproducibility over raw speed. (It takes about 30 minutes for a case)

## 🔧 Adding New Pipeline Steps

Each pipeline step follows the same structure in run.py:

```text
def step_new_module(ctx: PipelineContext) -> None:
    run_external_script(
        ctx,
        script_relative_path="Folder/NewModule.py",
    )
```
Optional skip check:
```text
def should_skip_new_module(ctx: PipelineContext) -> bool:
    output_path = Path(
        rf"C:\Users\User\Desktop\AbdVesselGen\Results\Intermediate\{ctx.case_id}_new_output.nii.gz"
    )
    return output_path.exists()
```
Then insert into:

```text
PIPELINE_STEPS = [
    ...
    PipelineStep(
        name="Step X - New Module",
        description="Description of the new module.",
        runner=step_new_module,
        enabled=True,
        skip_check=should_skip_new_module,
    ),
]
```
This modular structure allows new vessel branches, new anatomical anchors, and new post-processing steps to be added incrementally.

## ⚠️ Notes
This project is under active development.
The current implementation is rule-based and anatomy-guided.
It depends on the quality of upstream segmentation outputs.
Poor contrast timing, motion artifact, unusual anatomy, or segmentation failure may affect vessel tracing.
Manual inspection in 3D Slicer is recommended.
Some scripts currently assume a Windows-style project path and may require path modification for other platforms.
## ⚠️ Disclaimer

This project is intended for research and educational purposes only.

It is not a certified medical device and should not be used as the sole basis for clinical diagnosis, treatment planning, operative planning, or clinical decision-making.

All outputs should be reviewed by qualified medical professionals before any research or clinical interpretation.

## 📄 License

MIT License

## 🙌 Acknowledgements

This project builds on or interacts with several open-source tools and medical imaging platforms:

TotalSegmentator

3D Slicer
MONAI
Nibabel
pynrrd
SciPy
NumPy
Nilearn

Special thanks to the open-source medical imaging community for providing the tools that make anatomy-guided image processing workflows possible.

##💡 Future Work

Planned or possible future improvements include:
```text
batch processing support
cross-platform path handling
better anatomical variant handling
more robust vessel graph validation
automatic QA visualization
branch-level confidence scoring
semi-automatic manual correction tools
improved venous / arterial separation
quantitative vascular measurements
integration with radiology AI workflows
Citation
```
If you use this project in academic work, please cite this repository.

A formal citation will be added once a manuscript or preprint is available.
