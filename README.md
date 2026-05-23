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
