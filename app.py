"""
RAG Engine without LangChain
- Handles embedding, storage, retrieval, and Gemini response
"""

import os
import tempfile
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
import google.generativeai as genai


# ----------------------
# Gemini + Chroma Setup
# ----------------------

class RAGChatbot:
    def __init__(self, api_key, persist_dir="./chroma_store"):
        self.api_key = api_key
        self.persist_dir = persist_dir

        # Init Google Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Init embedding function (Google Embedding API)
        self.embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            model_name="models/embedding-001", api_key=self.api_key
        )

        # Init Chroma client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        # Create collection if not exists
        self.collection = self.chroma_client.get_or_create_collection(
            name="docs", embedding_function=self.embedding_fn
        )

        # Memory for conversation
        self.chat_history = []

    # ----------------------
    # Document ingestion
    # ----------------------
    def add_text(self, text, doc_id):
        self.collection.add(documents=[text], ids=[doc_id])

    def add_pdf(self, file_path):
        pdf = PdfReader(file_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        self.add_text(text, doc_id=os.path.basename(file_path))

    # ----------------------
    # Query with RAG
    # ----------------------
    def query(self, user_query, top_k=3):
        # Retrieve top-k docs
        results = self.collection.query(query_texts=[user_query], n_results=top_k)
        retrieved_docs = results["documents"][0]

        # Build context
        context = "\n\n".join(retrieved_docs)

        # Conversation history
        history = "\n".join([f"{role}: {msg}" for role, msg in self.chat_history])

        # Prompt
        prompt = f"""
You are a helpful AI assistant.
Conversation so far:
{history}

Relevant context:
{context}

User question: {user_query}

Answer in a helpful and concise way.
"""

        # Call Gemini
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        answer = response.candidates[0].content.parts[0].text

        # Save chat history
        self.chat_history.append(("User", user_query))
        self.chat_history.append(("Bot", answer))

        return answer
