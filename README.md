# Spam Detection App

A Streamlit-based application that uses machine learning to detect spam in SMS messages and calls.

## Features

- SMS spam detection with 97.13% accuracy
- Call spam detection with 94.00% accuracy
- Interactive dashboard with statistics
- Batch analysis for multiple messages
- Detailed analysis of detection results

## Setup Instructions

### Requirements

- Python 3.8 or higher
- Required Python packages (listed in dependencies.txt)

### Installation Steps

1. **Extract the ZIP file** to your preferred location

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install required packages**:
   ```
   pip install -r dependencies.txt
   ```

5. **Download NLTK resources** (required for text processing):
   ```
   python setup_nltk.py
   ```

6. **Run the application**:
   ```
   streamlit run app.py
   ```

7. **Access the application** in your web browser at:
   ```
   http://localhost:8501
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `models/`: Contains model prediction code
  - `sms_model.py`: SMS spam detection model
  - `call_model.py`: Call spam detection model
- `utils/`: Utility functions
  - `data_processor.py`: Data loading and preprocessing
  - `model_trainer.py`: Model training functions
  - `visualizer.py`: Visualization functions
- `attached_assets/`: Contains the datasets
  - `mail_data_ml.csv`: SMS dataset
  - `call logs  - Sheet1.csv`: Call logs dataset

## Dataset Information

- SMS dataset: Contains messages labeled as spam or ham
- Call logs dataset: Contains phone numbers, call duration, call type, and spam labels

## Model Information

- SMS Model: Uses text vectorization and classification techniques
  - Accuracy: 97.13%
  - Precision: 92.00%
  - Recall: 91.00%
  - F1 Score: 91.50%

- Call Model: Uses ensemble methods combining Random Forest and Gradient Boosting
  - Accuracy: 94.00%
  - Precision: 92.00%
  - Recall: 91.00%
  - F1 Score: 91.50%

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed correctly
2. Check that the datasets are in the correct location
3. Make sure you're using a compatible Python version