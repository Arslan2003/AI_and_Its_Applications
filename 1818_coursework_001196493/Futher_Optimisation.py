import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, auc
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
df = pd.read_csv('HI-Small_Trans_formatted.csv')

# Reshuffle the dataset
logging.info("Reshuffling the dataset")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle the entire dataset

# Reduce the dataset size by sampling
logging.info("Reducing the dataset size by sampling")
df = df.sample(frac=0.25, random_state=42).reset_index(drop=True)

# Convert categorical columns to numerical
logging.info("Converting categorical columns to numerical")
df = pd.get_dummies(df, columns=['Sent Currency', 'Received Currency', 'Payment Format'])

# Define features and target
X = df.drop(columns=['Is Laundering'])
y = df['Is Laundering']

# Split the data into 60% training and 40% temporary set
logging.info("Splitting the data into training and temporary sets")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Split the temporary set into 50% validation and 50% testing (20% each of the original data)
logging.info("Splitting the temporary set into validation and testing sets")
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
logging.info("Standardizing the features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
logging.info("Training the model with validation data")
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_val_scaled, y_val))

# Evaluate the model
logging.info("Evaluating the model on the test set")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print('Test accuracy:', test_acc)

# Make predictions on the test set
logging.info("Making predictions on the test set")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate evaluation metrics
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
confusion_mat = confusion_matrix(y_test, y_pred)

print("F1-Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", confusion_mat)

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
disp.plot()
plt.show()

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
