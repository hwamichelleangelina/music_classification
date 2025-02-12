# Music Genre Classification

This repository contains a **music genre classification** project that utilizes **feature extraction, data augmentation, and deep learning** to classify music into different genres.

Source of dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## Project Overview
The model is trained using a publicly available dataset of music audio .wav files. The approach utilizes deep learning techniques, specifically CNN, to analyze and classify audio files based on the genre of music. The system is designed to be mostly accurate, achieving an accuracy over 95% on training set, but 75% on testing datasets. This indicates overfitting in the model while training.

## Key Features
- **Feature Extraction**: Extracts MFCC, chroma, spectral contrast, and zero-crossing rate from audio files.
- **Data Augmentation**: Uses noise addition, pitch shifting, and time stretching, reverberation/echo, and time shifting to improve model robustness.
- **Deep Learning Model**: A **hybrid CNN-LSTM** architecture for effective audio sequence learning.
- **Preprocessing Pipeline**: Encodes labels and scales features for better training performance.
- **Training Optimizer** for improved training performance.
- **Overfitting Evaluation**: The model is evaluated to ensure there is no overfitting, and generalizes well to unseen data.

## Technologies
- Python
- Pydub and SoX
- Torchaudio
- NumPy
- SciPy
- Librosa
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib
- Tkinter
- IPython.display and ipywidgets
- Google Colab (for dataset access)

## Model Architecture
- **Conv1D** for feature extraction
- **LSTM** for sequence learning
- **Dense layers** for classification

## Project Directory
```
├── data
│   └── processed_data_2   # Folder for preprocessed data
├── src
│   ├── preprocess.py    # Feature extraction and augmentation
│   ├── train.py         # Model training script
│   ├── model.py         # CNN-LSTM model architecture
│   └── evaluate.py      # Model evaluation
├── music_genre_classification_notebook.ipynb
├── README.md
└── requirements.txt     # Dependencies
```

## Installation
To run this project locally, the steps below:

#### Step 1: Clone the repository
   ```bash
   git clone https://github.com/hwamichelleangelina/music_genre_classification.git
   cd music_genre_classification
   ```
#### Step 2: Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
Here’s a quick guide on the key libraries used:
`tensorflow, tensorflowjs, torchaudio, split-folders, scikit-learn, matplotlib`

#### Step 3: Download Dataset
To access the dataset, you need to use your Kaggle account. Upload the Kaggle API key file (kaggle.json) to this environment.

```
from google.colab import files
files.upload()
```
Once uploaded, the dataset is fetched and extracted using the Kaggle API

#### Step 4: Model Architecture
The model combines Convolutional Neural Networks (CNNs) for feature extraction and Bidirectional LSTMs for temporal sequence learning. The architecture consists of:

- Two Conv1D layers with increasing filters (64 and 128), followed by Batch Normalization and MaxPooling1D for dimensionality reduction.
- Two Bidirectional LSTM layers (128 and 64 units) to capture temporal dependencies.
- Flatten layer to convert the output into a 1D vector.
- Fully connected Dense layers, with:
1. 128 units and ReLU activation, followed by Dropout (0.5) for regularization.
2. An output layer using softmax activation to classify genres.

#### Step 5: Training and Optimization
The model is compiled with:

- AdamW optimizer (learning rate: 0.001) for adaptive learning.
- Categorical Crossentropy loss for multi-class classification.
- Accuracy as the evaluation metric.

To prevent overfitting, EarlyStopping (patience=5) and ReduceLROnPlateau (patience=3, factor=0.5) callbacks are applied. The model is trained for 50 epochs with 20% validation split.

#### Step 6: Performance Metrics
The trained model is evaluated on the test set using:

- Accuracy to measure overall correctness.
- Confusion Matrix to visualize misclassifications.
- F1 Score to evaluate precision and recall balance.
- Classification Report to provide per-class performance.

The results demonstrate the model’s ability to generalize across different music genres.

## Usage of Python Script
### 1️. Preprocess the Data
```bash
python src/preprocess.py
```

### 2. Train the Model
```bash
python src/train.py
```

### 3. Evaluate the Model
```bash
python src/evaluate.py
```

## Results
- **Accuracy**: The model achieved an accuracy 96% on training set, but 75% on testing dataset.
- **Overfitting Check**: The model has been evaluated for overfitting, and the results indicate that the model is overfitting.

## Future Improvement
- Fine-tuning the CNN-LSTM model
- Experimenting with other architectures (ResNet, CRNN)
- Implementing additional data augmentation techniques

## License
The Dataset source is licensed under the **CC0: Public Domain** license. You can use, modify, and distribute this code freely without any restrictions.
