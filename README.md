# M.I.N.D - Mental Interpretation Network for Decision Making

## Dataset Information
This analysis uses the **BCI Competition 2008 – Graz data set A** (also known as BCI Competition IV dataset 2a).

### Dataset Details
- **Subjects**: 9 subjects (A01T - A09T for training, A01E - A09E for evaluation)
- **Tasks**: 4-class motor imagery (Left hand, Right hand, Both feet, Tongue)
- **Channels**: 22 EEG channels + 3 EOG channels
- **Sampling rate**: 250 Hz
- **Trials**: 288 per subject (72 per class)
- **Runs**: 6 runs with short breaks

### Experimental Protocol
Each trial follows this timeline:
- **t = 0s**: Fixation cross + acoustic warning
- **t = 2s**: Visual cue appears (arrow direction)
- **t = 3.25s**: Cue disappears
- **t = 6s**: Trial ends

### File Naming Convention
- **AXXT.gdf**: Training data (use for classifier development)
- **AXXE.gdf**: Evaluation data (use for testing)

## Current Analysis (To Be Implemented)

This script will the following analyses:

1. **Preprocessing**
   - Bandpass filter: 7-35 Hz
   - Notch filter: 60 Hz (power line noise)

2. **Epoch Extraction**
   - Epochs from 0-4 seconds after cue (covers motor imagery period)
   - 288 trials extracted with proper class labels

3. **Evoked Response Analysis**
   - Grand average plots for each motor imagery task
   - Focus on motor cortex channels (C3, C4, Cz)

4. **Common Spatial Patterns (CSP)**
   - Spatial filters optimized for distinguishing left vs right hand
   - 5-fold cross-validation accuracy

5. **Classification**
   - Binary classification: Left vs Right hand
   - 4-class classification using CSP + SVM

6. **Time-Frequency Analysis**
   - Event-related desynchronization (ERD) patterns
   - Mu (8-12 Hz) and beta (13-30 Hz) rhythm analysis

## Results

Classification results for A09T:

*Note: Actual results may vary slightly depending on preprocessing parameters.*

## How to Download Other Subjects

### Option 1: Download from BCI Competition Website
Visit: http://www.bbci.de/competition/iv/
- Download "Dataset 2a" (Graz dataset A)
- Files: A01T.gdf through A09T.gdf (training)
- Files: A01E.gdf through A09E.gdf (evaluation)

### Option 2: Download via Python (check `download_data.py`)

### Option 3: Manual Download
1. Go to: http://www.bbci.de/competition/iv/
2. Register for a free account (required for dataset access)
3. Download "Dataset 2a" (approximately 2.5 GB total)
4. Extract all .gdf files to the same folder