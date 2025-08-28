# =============================== #
# Multimodal Housing Price Prediction
# Using Tabular Data + Images
# =============================== #

# ---- Install dependencies (only in Colab) ----
# !pip install scikit-learn pandas tensorflow

# ---- Imports ----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os

# ===============================
# STEP 1: Load CSV File
# ===============================
from google.colab import files
uploaded_csv = files.upload()   # Upload your CSV
csv_filename = list(uploaded_csv.keys())[0]

data = pd.read_csv(csv_filename)

print("CSV Loaded ✅")
print(data.head())

# ===============================
# STEP 2: Upload Images
# ===============================
uploaded_images = files.upload()   # Upload multiple house images

image_files = list(uploaded_images.keys())
print("Images uploaded ✅", image_files[:5])

# ===============================
# STEP 3: Pre-trained CNN for image features
# ===============================
cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_image_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

# ===============================
# STEP 4: Match Images with CSV Rows
# ===============================
image_features = []
for idx in range(len(data)):
    if idx < len(image_files):   # Assign by order
        feats = extract_image_features(image_files[idx], cnn_model)
    else:
        feats = np.zeros(2048)  # Fallback if missing image
    image_features.append(feats)

image_features = np.array(image_features)

print("Extracted Image Features ✅", image_features.shape)

# ===============================
# STEP 5: Prepare Tabular Data
# ===============================
# Target = price
y = data["price"].values  

# Drop non-numeric columns
X_tab = data.drop(columns=["price"])
X_tab = pd.get_dummies(X_tab, drop_first=True)  # Encode categorical

# Normalize
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab)

print("Tabular Features ✅", X_tab_scaled.shape)

# ===============================
# STEP 6: Combine Tabular + Image Features
# ===============================
X_final = np.hstack([X_tab_scaled, image_features])
print("Final Shape:", X_final.shape)

# ===============================
# STEP 7: Train & Evaluate Model
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Training ✅")
print("MSE:", mean_squared_error(y_test, y_pred))
