from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import logging
from groq import Groq
from pathlib import Path
import tempfile
from datetime import datetime
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key")
client = Groq(api_key=GROQ_API_KEY)

HISTORY_FILE = "chat_history.json"
chat_history = []
pdf_vector_store = None
pdf_uploaded = False

def save_chat_history():
    try:
        with open(HISTORY_FILE, "w", encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

def load_chat_history():
    global chat_history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding='utf-8') as f:
                chat_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")

load_chat_history()

def initialize_rag_pipeline(pdf_path: str):
    global pdf_vector_store, pdf_uploaded
    try:
        # Extract text from PDF
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        
        if not text.strip():
            raise ValueError("No text extracted from PDF")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        pdf_vector_store = FAISS.from_texts(chunks, embeddings)
        pdf_uploaded = True
        logger.debug("RAG pipeline initialized with PDF")
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def generate_response(query: str, from_pdf: bool = False) -> str:
    try:
        if from_pdf and pdf_vector_store:
            # PDF agent: Use RAG pipeline
            prompt_template = PromptTemplate(
                template="Use the following context to answer the question concisely:\n{context}\n\nQuestion: {question}\nAnswer:",
                input_variables=["context", "question"]
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=client,
                chain_type="stuff",
                retriever=pdf_vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt_template}
            )
            response = qa_chain.run({"question": query})
        else:
            # General agent: Use Groq chat
            messages = [
                {"role": "system", "content": "You are Grok, a helpful AI assistant created by xAI. Respond in a friendly, conversational tone."}
            ]
            for msg in chat_history[-4:]:
                messages.append({"role": "user", "content": msg["query"]})
                messages.append({"role": "assistant", "content": msg["response"]})
            messages.append({"role": "user", "content": query})
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            response = response.choices[0].message.content.strip()
        return response
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return f"Oops, something went wrong: {str(e)}. Try again!"

class TextQuery(BaseModel):
    query: str
    from_pdf: bool = False

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding='utf-8') as f:
            return f.read()
    return HTMLResponse("<h1>Voice Chatbot</h1><p>Please add index.html to static/</p>")

@app.get("/chat_history")
async def get_chat_history():
    return chat_history

@app.get("/pdf_status")
async def get_pdf_status():
    return {"pdf_uploaded": pdf_uploaded}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_vector_store, pdf_uploaded
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    temp_file_path = None
    try:
        logger.debug(f"Received PDF file: {file.filename}, content_type: {file.content_type}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        initialize_rag_pipeline(temp_file_path)
        return {"message": "PDF uploaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Removed temp file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file_path}: {e}")

@app.post("/process_voice")
async def process_voice(audio: UploadFile = File(...)):
    temp_file_path = None
    try:
        logger.debug(f"Received audio file: {audio.filename}, content_type: {audio.content_type}")
        
        file_extension = os.path.splitext(audio.filename)[1].lower() or '.webm'
        if not file_extension:
            content_type = audio.content_type.lower()
            if 'webm' in content_type:
                file_extension = '.webm'
            elif 'mp4' in content_type or 'mpeg' in content_type:
                file_extension = '.mp4'
            elif 'ogg' in content_type:
                file_extension = '.ogg'
            else:
                file_extension = '.webm'

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.debug(f"Temporary file saved: {temp_file_path}")

        try:
            os.close(temp_file.fileno())
        except:
            pass

        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio.filename or f"recording{file_extension}", audio_file),
                model="whisper-large-v3"
            )
            query = transcription.text.strip()
            logger.debug(f"Transcribed query: {query}")
        
        if not query:
            raise HTTPException(status_code=400, detail="No speech detected")
        
        from_pdf = pdf_uploaded and any(keyword in query.lower() for keyword in ["pdf", "document", "file"])
        response = generate_response(query, from_pdf)
        add_to_chat_history("voice", query, response)
        return {"query": query, "response": response}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Removed temp file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file_path}: {e}")

@app.post("/process_text")
async def process_text(query: TextQuery):
    try:
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Empty text input")
        logger.debug(f"Received text query: {query.query}, from_pdf: {query.from_pdf}")
        response = generate_response(query.query, query.from_pdf)
        add_to_chat_history("text", query.query, response)
        return {"query": query.query, "response": response}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.delete("/clear_history")
async def clear_chat_history():
    global chat_history, pdf_vector_store, pdf_uploaded
    chat_history = []
    pdf_vector_store = None
    pdf_uploaded = False
    save_chat_history()
    return {"message": "Chat history cleared"}

if __name__ == "__main__":
    import uvicorn
    os.makedirs("static", exist_ok=True)
    uvicorn.run(app, port=8000)
