import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load dataset
X_test = np.load("data/processed_data_2/X_test.npy")
y_test = np.load("data/processed_data_2/y_test.npy")

# Load model
model = load_model("music_genre_classifier.h5")

# Evaluasi Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Prediksi
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# F1 Score
f1 = f1_score(y_test_labels, Y_pred_classes, average='macro')
print(f'F1 Score (Macro): {f1:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, Y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(set(y_test_labels)),
            yticklabels=sorted(set(y_test_labels)))

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
print(classification_report(y_test_labels, Y_pred_classes, target_names=[str(i) for i in sorted(set(y_test_labels))]))