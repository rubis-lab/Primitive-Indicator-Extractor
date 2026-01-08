# Primitive-Indicator-Extractor

Primitive-Indicator-Extractor is a collection of scripts that extract primitive indicators (PIs) from sensor data. The supported modalities are eye-tracking, touch, keyboard, and voice.

## Scripts

PI extraction logic is implemented as separate Python scripts per modality:

- `extract_eye-tracking_pi.py`
- `extract_touch_pi.py`
- `extract_keyboard_pi.py`
- `extract_voice_pi.py`

Each script is executed via a corresponding shell runner:

- `run_eye_pi.sh`
- `run_touch_pi.sh`
- `run_keyboard_pi.sh`
- `run_voice_pi.sh`

## Usage

### Requirements
- Conda (recommended)
- Python 3.11

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate primitive-indicator-extractor

