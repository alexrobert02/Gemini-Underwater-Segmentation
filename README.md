# Underwater Saliency Detection with Gemini 2.5 Pro

This project uses the Gemini 2.5 Pro vision model to perform saliency detection on underwater images. The script identifies key objects (fish, reefs, divers, etc.) and exports the data into a structured JSON format suitable for computer vision tasks.

## Overview
Underwater object detection is notoriously difficult due to poor lighting, color distortion, and turbidity. While specialized models like **USIS-SAM** use adapter modules to fine-tune foundational models for these conditions, this project explores a different approach: **Can a general-purpose Multimodal LLM (Gemini 2.5 Pro) compete as a zero-shot underwater analyst?**

We integrated the Gemini API to identify salient objects, generate bounding boxes, and classify them into 7 strict marine categories.

## Methodologies
### 1. Model & Prompting
We utilized **Gemini 2.5 Pro** acting as an expert domain analyst.
* **Role:** Expert Underwater Image Analyst.
* **Task:** Identify the most "salient" (visually prominent) objects.
* **Output:** For every detected object, the model returns:
   * `label`: The high-level category.
   * `detailed_label`: A specific description (e.g., "Brain Coral").
   * `box_2d`: Normalized 2D Bounding Boxes `[ymin, xmin, ymax, xmax]` (scale 0-1000)

### 2. Category Mapping
The model was restricted to the 7 specific categories defined by the USIS10K standard:

| Category       | Descriptions                                |
|----------------|---------------------------------------------|
| Fish           | Underwater vertebrates, e.g., fish, turtles |
| Reefs          | Underwater invertebrates and coral reefs    |
| Aquatic plants | Aquatic plants and flora                    |
| Wrecks/ruins   | Wrecks, ruins and damaged artifacts         |
| Human divers   | Human divers and their equipment            |
| Robots         | Underwater robots like AUV, ROV             |
| Sea-floor      | Rocks and reefs on the seafloor             |

### 3. Robust API Integration
To handle large-scale processing, we implemented:
* **Exponential Backoff:** A retry mechanism (up to 5 attempts) to handle `Resource exhausted / 429 Too Many Requests` errors.
* **Batch Processing:** Logic to manage daily quota limits and segmented execution.

## Dataset Details
This project heavily relies on the `USIS10K Dataset`, a large-scale underwater dataset introduced by the authors of the USIS-SAM paper.

* **Dataset Name:** USIS10K (Underwater Salient Instance Segmentation 10K)
* **Original Repository:** [USIS10K](https://github.com/LiamLian0727/USIS10K)
* **Primary Contribution:** The dataset addresses the critical lack of labeled training data for underwater environments, providing 10,632 images with pixel-level annotations across diverse scenes like coral reefs and shipwrecks.

## Result
We benchmarked Gemini 2.5 Pro against the state-of-the-art **USIS-SAM** and the general **SAM3 model**. Due to the subjectivity of "saliency" in the ground truth, we prioritized **Recall** (the ability to find the correct objects).

### Key Findings (Label Agnostic Recall)
| Model               | Recall | Notes                            |
|---------------------|--------|----------------------------------|
| USIS-SAM (Baseline) | 87.9%  | Specialized, fine-tuned model    |
| Gemini 2.5 Pro      | 78.9%  | Impressive Zero-shot performance |
| SAM3 (Simple)       | 60.4%  | General purpose segmentation     |
| SAM3 (Descriptive)  | 44.4%  | Struggled with complex prompts   |

### Class-Specific Competitiveness
Gemini 2.5 Pro showed it can rival specialized models in distinct categories:
* **Human Divers:** 92.7% Recall (vs 97.4% USIS-SAM)
* **Fish:** 88.8% Recall (vs 92.5% USIS-SAM)
* **Wrecks/Ruins:** 84.3% Recall (vs 91.5% USIS-SAM)

## Challenges & Limitations
* **Saliency Ambiguity:** Defining what is "salient" underwater is subjective. In many cases, the definition of a salient object in the Ground Truth is unclear (e.g., empty backgrounds vs. specific fish).
* **API Constraints:** Large-scale evaluation is constrained by API rate limits and daily quotas, necessitating complex retry logic.
* **Subjective Ground Truth:** Visual inspection revealed cases where Gemini detected valid objects that were missing from the Ground Truth annotations.

## Credits
**Attribution:** This repository uses the USIS10K dataset for benchmarking and evaluation purposes only. We claim no ownership over the data. All credit for the collection, annotation, and curation belongs to the original authors.

**Original Paper:** [Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset](https://proceedings.mlr.press/v235/lian24c.html)

