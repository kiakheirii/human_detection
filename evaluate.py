import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
from preprocessing import test_dir as ts

# Load test dataset properly
test_dir = ts # Ensure this is the correct path

test_data = image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),  # Match model input size
    batch_size=32,
    shuffle=False  # Important for correct label order
)

# Get class labels
class_names = test_data.class_names  # This gives class labels
y_true = np.concatenate([y.numpy() for x, y in test_data])  # Extract true labels

# Load trained model
model = load_model('cnn_model_trained.h5')

# Predict class indices
y_pred = np.argmax(model.predict(test_data), axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
