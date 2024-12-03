# biomedical-alvaroLLM

ALVARO-LLM is a framework designed to integrate multimodal data from brain MRI and clinical reports to aid in the classification of cerebrovascular conditions, particularly Cerebral Cavernous Malformations (CCMs) and Acute Intraparenchymal Hemorrhage (AIHs). 


## Architecture

- Text Analysis: Extract insights from clinical reports using BERT.
- MRI Processing: Use NiBabies for motion correction, normalization, and brain extraction.
- Fusion Architecture: Combines textual and imaging features for joint classification.
- Evaluation Metrics: AUC-ROC, F1-score, and slice/patient-level accuracy.

## Installation Steps
1. Clone repository:

```bash
git clone https://github.com/your-repo/ALVARO-LLM.git
cd ALVARO-LLM
```

2. Setup Environment

```bash
conda create -n alvaro-llm python=3.8
conda activate alvaro-llm
pip install -r requirements.txt
```

3. Download Pretrained Models:
Use transformers for BERT and MONAI for 3D CNNs.

4. Preprocess MRI Data:
Run `NiBabies preprocessing` on your BIDS-compliant dataset:

## Usage
1. Train the model
```bash
python train_alvaro.py --text_path data/reports --mri_path path/to/output_directory
```

2. Evaluate Alvaro-LLM
```bash
python evaluate_alvaro.py --test_path path/to/test_data
```

