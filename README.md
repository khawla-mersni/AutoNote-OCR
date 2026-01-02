# AutoNote-OCR

**Automatic Exam Score Extraction Using Computer Vision and Explicit Fuzzy Logic**

---

## Project Overview

**AutoNote-OCR** is an intelligent computer vision system designed to automatically extract exam scores from scanned answer sheets.  
The system focuses on robust score detection under real-world conditions such as scanning noise, uneven illumination, rotation, partial occlusions, and ambiguous markings.

Unlike purely threshold-based approaches, this project integrates an **explicit fuzzy logic decision layer** to formally model uncertainty and provide **confidence-aware results**, making it suitable for academic evaluation and automated grading systems.

---

## Objectives

The main objectives of this project are:

- Automatically locate and segment score selection grids from scanned exam sheets  
- Detect both integer and decimal score components  
- Handle ambiguous, weak, or noisy markings robustly  
- Quantify the reliability of each extracted result using fuzzy logic  
- Produce structured, auditable outputs for validation and reporting  

---

## Technical Approach

The proposed system combines **classical computer vision techniques** with **explicit fuzzy logic reasoning**.

The design philosophy prioritizes:
- Deterministic and explainable decisions  
- Robustness to acquisition variability  
- Avoidance of black-box machine learning models  
- Academic-grade traceability and reproducibility  

---

## Processing Pipeline

### 1. Image Acquisition and Orientation Correction
- Input images: scanned exam copies (JPEG / PNG)
- Orientation correction using QR code detection
- Automatic handling of inverted pages (180° rotation)

### 2. Relative Region of Interest (ROI) Extraction
- Score grids are extracted using **relative coordinates**
- Ensures robustness to image resolution and scaling variations
- Eliminates dependency on absolute pixel positions

### 3. Adaptive Thresholding
- Local adaptive thresholding is applied
- Handles non-uniform lighting and background variations
- Converts grayscale images into binary representations

### 4. Morphological Processing
- **Opening (3×3 kernel)**: removes small noise and isolated pixels
- **Closing (5×5 kernel)**: reconnects broken grid lines and structures
- Kernel sizes are empirically selected to preserve grid integrity

### 5. Grid Detection Using Projection Profiles
- Horizontal and vertical projections of binary images
- Peak detection and grouping to identify grid lines
- Automatic separation between integer and decimal sub-grids

### 6. Cell Segmentation
- Grid intersections define individual score cells
- Each cell is processed independently
- Only the lower half of each cell is analyzed to avoid printed text interference

---

## Fuzzy Logic Integration

### Implicit Fuzzy Reasoning

Initially, uncertainty is handled implicitly using:
- Relative comparisons between ink densities
- Adaptive thresholds based on local statistics
- Heuristic decision rules

Although effective, this approach does not explicitly model uncertainty.

---

### Explicit Fuzzy Logic Model

To formally represent uncertainty, an **explicit fuzzy logic system** is introduced.

#### Fuzzy Variables
- **Mark Strength**: relative amount of detected ink in a cell  
- **Contrast Gap**: difference between the strongest and second strongest candidates  
- **Candidate Count**: number of plausible marked cells  

#### Membership Functions
Each variable is mapped to fuzzy sets such as:
- *Low*, *Medium*, *High*

Membership degrees are continuous values in the range \([0, 1]\).

#### Fuzzy Rules
Examples of decision rules include:
- IF mark strength is *High* AND contrast gap is *Large* → Confidence is *High*  
- IF two candidates are close → Confidence is *Low*  
- IF too many candidates exist → Result is *Invalid*  

#### Defuzzification
- Fuzzy outputs are aggregated using weighted combinations
- The final output is a **confidence score in the range [0, 1]**
- The score represents reliability, not correctness

---

## Confidence Score Interpretation

For each detected score, the system computes:

- `confidence_int`: confidence in integer selection  
- `confidence_dec`: confidence in decimal selection  
- `confidence_global`: aggregated confidence score  

### Interpretation Guide

| Confidence Range | Interpretation |
|------------------|----------------|
| 0.80 – 1.00 | Highly reliable |
| 0.60 – 0.80 | Acceptable, low ambiguity |
| 0.40 – 0.60 | Uncertain, manual verification recommended |
| < 0.40 | Unreliable, rejected |

This mechanism enables informed decision-making and result auditing.

---

## Output Format

The system generates a structured Excel file containing:
- filename  
- grid validity status  
- detected score  
- integer and decimal indices  
- confidence scores  
- error flags  

This format is suitable for evaluation, reporting, and validation workflows.

---

## Repository Structure
AutoNote-OCR/
├── main.py # Core processing pipeline
├── data/ # Input images and outputs
├── docs/ # Technical documentation
├── README.md
├── LICENSE



##Design Philosophy

Explainable and deterministic algorithms

Robustness over overfitting

Explicit uncertainty modeling

Academic and professional-grade implementation



##License

This project is released under the MIT License.



## How to Run

```bash
pip install -r requirements.txt
python main.py```



