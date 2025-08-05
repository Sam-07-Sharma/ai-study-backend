import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import PyPDF2
from pptx import Presentation
import io
import logging
import sys
import json
import uuid 
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

app = Flask(__name__)

# In-memory cache to store content for different sessions
document_cache = {}

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Functions for Text Extraction ---
def extract_text_from_pdf(pdf_file_stream):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file_stream)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None

def extract_text_from_pptx(pptx_file_stream):
    text = ""
    try:
        prs = Presentation(pptx_file_stream)
        text = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
        return text
    except Exception as e:
        logger.error(f"PPTX extraction error: {e}")
        return None

# --- Helper Functions for AI Processing ---
def get_ai_response(prompt, document_text=None, image_bytes=None, mime_type=None):
    """Handles all communication with the Gemini API."""
    try:
        parts = []
        
        # Add context if it exists
        if document_text:
            context_prompt = (
                "You are an AI study assistant. Use the provided document context to answer the user's query. "
                "You can summarize, analyze, and generate new content like questions based on the information within the document.\n\n"
                f"--- DOCUMENT CONTEXT ---\n{document_text}\n\n"
                f"--- USER QUERY ---\n{prompt}"
            )
            parts.append({"text": context_prompt})
        elif image_bytes:
            context_prompt = (
                "You are an AI study assistant. Analyze the following image and answer the user's query about it.\n\n"
                f"--- USER QUERY ---\n{prompt}"
            )
            import base64
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            parts.append({"text": context_prompt})
            parts.append({"inline_data": {"mime_type": mime_type, "data": encoded_image}})
        else:
            # No context, just a general query
            parts.append({"text": prompt})

        payload = {"contents": [{"parts": parts}]}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result_json = response.json()
        if 'candidates' in result_json and result_json['candidates']:
            return result_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return "The AI could not generate a response due to safety settings."

    except Exception as e:
        logger.error(f"AI generation error: {e}")
        return "Sorry, an error occurred while communicating with the AI."

# --- API Endpoints ---

@app.route('/ai-query', methods=['POST'])
def ai_query():
    """A single endpoint to handle file uploads and queries."""
    
    # Case 1: File Upload
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()
        file_bytes = file.read()
        file_stream = io.BytesIO(file_bytes)
        session_id = str(uuid.uuid4())

        # Handle images
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        if file_extension in image_extensions:
            try:
                Image.open(file_stream)
                mime_type = f'image/{file_extension[1:]}'
                document_cache[session_id] = {'type': 'image', 'content': file_bytes, 'mime_type': mime_type}
                return jsonify({"session_id": session_id, "message": "Image processed. You can now ask questions about it."})
            except Exception as e:
                return jsonify({"error": "Uploaded file is not a valid image."}), 400

        # Handle documents
        text = None
        if file_extension == '.pdf': text = extract_text_from_pdf(file_stream)
        elif file_extension == '.pptx': text = extract_text_from_pptx(file_stream)
        elif file_extension == '.txt': text = file_bytes.decode('utf-8')
        else: return jsonify({"error": "Unsupported document type"}), 400

        if not text or not text.strip():
            return jsonify({"error": "Failed to extract text from the document."}), 500

        document_cache[session_id] = {'type': 'text', 'content': text}
        return jsonify({"session_id": session_id, "message": "File processed. You can now ask questions about it."})

    # Case 2 & 3: Query (with or without context)
    data = request.get_json()
    query = data.get('query')
    session_id = data.get('session_id') # This will be null for general queries

    if not query:
        return jsonify({"error": "Query is required."}), 400

    ai_response = ""
    if session_id:
        # Query with context
        cached_data = document_cache.get(session_id)
        if not cached_data:
            return jsonify({"error": "Invalid session ID. Please upload the document again."}), 404
        
        content_type = cached_data.get('type')
        content = cached_data.get('content')

        if content_type == 'text':
            ai_response = get_ai_response(query, document_text=content)
        elif content_type == 'image':
            mime_type = cached_data.get('mime_type')
            ai_response = get_ai_response(query, image_bytes=content, mime_type=mime_type)
    else:
        # General query without context
        ai_response = get_ai_response(query)
    
    return jsonify({"response": ai_response})

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
