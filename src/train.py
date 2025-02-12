import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import build_model

# Load dataset
X = np.load("data/processed_data_2/X_features.npy")
y = np.load("data/processed_data_2/y_labels.npy")

# Preprocessing
X = X.reshape(X.shape[0], 40, -1)
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model building
input_shape = (40, X_train.shape[2])
model = build_model(input_shape, num_classes)
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1)

# Training
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stopping, lr_reduction]
)

# Save the trained model
model.save("music_genre_classifier.h5")