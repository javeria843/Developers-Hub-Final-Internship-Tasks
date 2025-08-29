📦 Task 2: End-to-End ML Pipeline – Telco Churn Prediction
Objective Build a reusable ML pipeline using Scikit-learn to predict customer churn from Telco data.

Dataset Telco Churn Dataset (CSV format)

Steps

Preprocessing: Scaling, encoding, missing value handling via Pipeline

Models: Logistic Regression, Random Forest

Tuning: GridSearchCV for hyperparameters

Export: Save final pipeline using joblib

How to Run

bash
python churn_pipeline.py
Output

Trained model pipeline (model.joblib)

Accuracy, precision, recall scores

GridSearchCV best parameters

Skills Gained

Scikit-learn Pipeline API

Hyperparameter tuning
<img width="1184" height="487" alt="image" src="https://github.com/user-attachments/assets/c1766b0b-83e0-4770-ba0c-8776e0df3b5f" />


🏠 Task 3: Multimodal ML – Housing Price Prediction
Objective Predict house prices using both tabular features and house images.

Dataset

Tabular: Housing Sales CSV

Images: House photos (JPEG/PNG)

Steps

CNN: Extract image features using pretrained model (e.g., ResNet)

Tabular: Normalize and encode structured data

Fusion: Concatenate image + tabular features

Model: Train regression model (e.g., XGBoost or MLP)

Evaluation: MAE, RMSE

How to Run

bash
python multimodal_housing.py
Output

Final model predictions

MAE and RMSE scores

Feature fusion pipeline

Skills Gained

Multimodal ML (image + tabular)

CNN feature extraction

Regression modeling

Performance evaluation

<img width="488" height="491" alt="image" src="https://github.com/user-attachments/assets/80ba07ce-f79c-40c0-b037-b00d34af29c4" />

🤖 Task 4: Context-Aware Chatbot with LangChain or RAG
Objective Build a chatbot that remembers conversation history and retrieves answers from external documents.

Dataset Custom corpus (Wikipedia, PDFs, internal docs)

Steps

Embedding: Convert documents to vectors using sentence-transformers

Vector Store: Store in FAISS or Chroma

Retrieval: Use LangChain or RAG to fetch relevant chunks

Memory: Enable chat history tracking

Deployment: Build UI with Streamlit

How to Run

bash
streamlit run chatbot_app.py
Output

Chatbot with memory + retrieval

Streamlit interface

Real-time responses from corpus

Skills Gained

LangChain / RAG integration

Semantic search via embeddings

Conversational memory

LLM deployment with Streamlit


Model export for production

Clean, modular ML code
<img width="1334" height="612" alt="image" src="https://github.com/user-attachments/assets/f5409650-7409-4293-8374-8342ba870783" />


