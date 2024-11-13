import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
import json
from typing import List, Dict, Any, Tuple
import shutil
import chromadb
from chromadb.config import Settings
import time
import threading
import uuid
from app import db
import re
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_thread_local = threading.local()

# Initialize cross-encoder for reranking
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    logger.error(f"Error initializing cross-encoder: {str(e)}")
    cross_encoder = None


def get_chroma_client(path: str) -> chromadb.PersistentClient:
    if not hasattr(_thread_local, 'clients'):
        _thread_local.clients = {}

    if path not in _thread_local.clients:
        settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            allow_reset=True
        )
        _thread_local.clients[path] = chromadb.PersistentClient(
            path=path,
            settings=settings
        )

    return _thread_local.clients[path]


def analyze_document_structure(text: str) -> Dict[str, Any]:
    """Analyze document structure to determine optimal processing strategy."""
    return {
        'has_numbered_lists': bool(re.search(r'^\s*\d+\.', text, re.MULTILINE)),
        'has_hierarchical_structure': bool(re.search(r'^[A-Z][^.!?]*[:.]', text, re.MULTILINE)),
        'average_sentence_length': len(text.split()) / max(len(text.split('.')), 1),
        'contains_tables': '[table]' in text.lower() or '|' in text,
    }


def process_document_for_agent(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text_content = []

        # Process each page with enhanced text extraction
        for page_num, page in enumerate(doc):
            # Get text blocks with preserved structure
            blocks = page.get_text("dict")["blocks"]
            page_text = []

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            # Preserve formatting and structure
                            if span.get("size", 0) > 12:  # Likely headers
                                line_text += f"\n{span['text']}\n"
                            else:
                                line_text += span['text'] + " "
                        page_text.append(line_text.strip())

            # Join text while preserving structure
            processed_text = "\n".join(filter(None, page_text))

            # Add page separator for better structure preservation
            if processed_text.strip():
                text_content.append(f"{processed_text}\n")

        # Create full text while preserving document structure
        full_text = "\n".join(text_content)

        # Check if we got meaningful content
        if len(full_text.strip()) < 100:  # Arbitrary minimum length
            # Fallback to raw text extraction
            text_content = []
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    text_content.append(text)
            full_text = "\n".join(text_content)

            # If still no good content, try OCR
            if len(full_text.strip()) < 100:
                return process_document_ocr(file_path)

        return full_text

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Fallback to OCR if regular extraction fails
        return process_document_ocr(file_path)


def process_document_ocr(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text_content = []
        languages = "deu+eng"
        config = f"--oem 3 --psm 6 -l {languages}"

        for page_num, page in enumerate(doc):
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increased resolution
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img, lang=languages, config=config)
                if text.strip():
                    text_content.append(text)
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                continue

        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        return ""


def classify_prompt(prompt: str, agent_description: str, agent_domain: str) -> Tuple[bool, str]:
    system_message = """You are a strict domain classifier for an AI agent. Analyze if the question matches the agent's expertise and document knowledge exactly.

    Classification rules:
    1. Return "true" ONLY if the question directly matches the agent's domain and documented knowledge
    2. Be extremely strict - reject anything outside the specific domain
    3. Consider both technical accuracy and scope
    4. Match the response language to the user's question
    5. Require exact domain alignment"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"""Agent Description: {agent_description}
        Domain Keywords: {agent_domain}
        Question: {prompt}

        Return JSON format:
        {{"is_relevant": boolean, "explanation": "explanation"}}"""}
    ]

    try:
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            groq_api_key=Config.GROQ_API_KEY
        )
        response = llm.invoke(messages)
        result = json.loads(response.content)
        return result["is_relevant"], result["explanation"]
    except Exception as e:
        logger.error(f"Error in prompt classification: {str(e)}")
        return False, "Classification failed"


def load_documents(agent) -> bool:
    store_dir = None
    try:
        store_dir = f'document_stores/agent_{agent.id}'
        os.makedirs(store_dir, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        all_texts = []
        doc_dir = f'data/agent_{agent.id}'

        if not os.path.exists(doc_dir):
            logger.error(f"Document directory not found: {doc_dir}")
            return False

        # Enhanced document processing
        for filename in os.listdir(doc_dir):
            file_path = os.path.join(doc_dir, filename)
            if filename.lower().endswith('.pdf'):
                text_content = process_document_for_agent(file_path)
                if text_content:
                    # Split content into semantic chunks
                    chunks = split_into_semantic_chunks(text_content)
                    all_texts.extend(chunks)

        if not all_texts:
            logger.error("No text content extracted from documents")
            return False

        # Create vector store
        try:
            if os.path.exists(store_dir):
                shutil.rmtree(store_dir)
            os.makedirs(store_dir)

            chroma_client = get_chroma_client(store_dir)
            collection_name = f"agent_{agent.id}_{uuid.uuid4().hex[:8]}"
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"agent_id": str(agent.id)}
            )

            # Process in smaller batches
            batch_size = 32
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_texts))]
                batch_embeddings = embeddings.embed_documents(batch_texts)

                collection.add(
                    documents=batch_texts,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )

            # Update agent configuration
            config = json.loads(agent.configurations)
            config['collection_name'] = collection_name
            agent.configurations = json.dumps(config)
            agent.document_store = store_dir
            db.session.commit()

            return True

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            if os.path.exists(store_dir):
                shutil.rmtree(store_dir)
            raise

    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        if store_dir and os.path.exists(store_dir):
            shutil.rmtree(store_dir)
        return False


def split_into_semantic_chunks(text: str) -> List[str]:
    """Split text into semantic chunks while preserving lists and sections."""
    chunks = []

    # Split text into major sections first
    sections = text.split("\nChapter")

    for section in sections:
        # Find lists (numbered or bulleted)
        list_pattern = r'(?:\d+\.|\•|\-)\s+[^\n]+(?:\n(?:\d+\.|\•|\-)\s+[^\n]+)*'
        lists = re.finditer(list_pattern, section, re.MULTILINE)

        # Keep track of processed positions
        last_pos = 0
        current_chunk = []

        for match in lists:
            # Add text before the list if any
            if match.start() > last_pos:
                pre_list_text = section[last_pos:match.start()].strip()
                if pre_list_text:
                    current_chunk.append(pre_list_text)

            # Add the complete list as one chunk
            list_text = match.group(0).strip()
            if list_text:
                # If we find a numbered list with a header/title before it
                pre_list_lines = section[max(0, match.start() - 200):match.start()].split('\n')
                list_title = next((line for line in reversed(pre_list_lines)
                                   if line.strip() and not line.strip().startswith(('•', '-', '*', '1'))), '')

                if list_title:
                    list_text = f"{list_title.strip()}\n{list_text}"

                chunks.append(list_text)

            last_pos = match.end()

        # Add remaining text
        if last_pos < len(section):
            remaining = section[last_pos:].strip()
            if remaining:
                current_chunk.append(remaining)

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

    # Post-process chunks to ensure they're not too large
    max_chunk_size = 2000
    processed_chunks = []

    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            # Split large chunks while preserving list integrity
            if any(line.strip().startswith(('1.', '•', '-')) for line in chunk.split('\n')):
                processed_chunks.append(chunk)  # Keep lists intact
            else:
                sentences = chunk.split('. ')
                current = []
                current_size = 0

                for sentence in sentences:
                    if current_size + len(sentence) > max_chunk_size:
                        processed_chunks.append('. '.join(current) + '.')
                        current = [sentence]
                        current_size = len(sentence)
                    else:
                        current.append(sentence)
                        current_size += len(sentence)

                if current:
                    processed_chunks.append('. '.join(current) + '.')
        else:
            processed_chunks.append(chunk)

    return processed_chunks


def create_list_specific_query(query: str) -> List[str]:
    """Create variations of queries specifically for finding lists in text."""
    base_query = query.lower()
    terms = []

    # Remove common question words and get core terms
    for word in ['what are', 'list', 'tell me', 'the', 'all', 'of']:
        base_query = base_query.replace(word, '')
    base_query = base_query.strip()

    return [
        f"1. {base_query}",  # Match numbered list start
        f"These are {base_query}",  # Match introductory phrase
        base_query,  # Original term
        f"{base_query} 1.",  # Match list content
        f"{base_query} are",  # Match definitional content
    ]

    # In process_query function, modify the query processing:
    if "list" in query.lower() or "what are" in query.lower():
        query_variations = create_list_specific_query(query)
        # Increase n_results for lists to ensure we get all items
        n_results = 15
    else:
        query_variations = [query]
        n_results = 8

    all_results = []
    for q in query_variations:
        query_embedding = embeddings.embed_query(q)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        all_results.extend(results['documents'][0])


def split_numbered_lists(text: str) -> List[str]:
    """Split text while preserving numbered list integrity."""
    chunks = []
    current_chunk = []
    current_size = 0

    # Split by potential list items
    items = re.split(r'(?=^\s*\d+\.)', text, flags=re.MULTILINE)

    for item in items:
        if current_size + len(item) > 1500:  # Max chunk size
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = [item]
            current_size = len(item)
        else:
            current_chunk.append(item)
            current_size += len(item)

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


def process_query(agent, query: str, conversation_history: List = None) -> str:
    try:
        config = json.loads(agent.configurations)
        collection_name = config.get('collection_name')
        model = config.get('model')
        temperature = float(config.get('temperature'))

        if not collection_name or not agent.document_store or not os.path.exists(agent.document_store):
            return "Agent initialization error"

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        try:
            chroma_client = get_chroma_client(agent.document_store)
            collection = chroma_client.get_collection(collection_name)

            # Get initial results with original query
            query_embedding = embeddings.embed_query(query)
            initial_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            # Get surrounding context by using nearby text
            all_results = initial_results['documents'][0]

            # If we found some results, try to get more context
            if all_results:
                # Extract key phrases from the initial results
                key_phrases = [sent.strip() for result in all_results
                               for sent in result.split('.') if len(sent.strip()) > 20]

                # Use these phrases to get additional context
                for phrase in key_phrases[:2]:  # Limit to avoid too many queries
                    context_embedding = embeddings.embed_query(phrase)
                    context_results = collection.query(
                        query_embeddings=[context_embedding],
                        n_results=3
                    )
                    all_results.extend(context_results['documents'][0])

            # Remove duplicates while preserving order
            seen = set()
            unique_results = []
            for result in all_results:
                normalized = ' '.join(result.split())
                if normalized not in seen:
                    seen.add(normalized)
                    unique_results.append(result)

            if not unique_results:
                return "I couldn't find relevant information in the provided document. Please rephrase your question or verify that this information exists in the uploaded content."

            # Join results while preserving any existing structure
            context = '\n'.join(unique_results)

            system_message = f"""You are a specialized AI assistant for {agent.description}.

            Requirements:
            1. Use ONLY the information from the provided context
            2. Present information exactly as it appears in the document
            3. If the exact information isn't in the context, clearly state that
            4. Preserve any structural elements (like lists or numbering) exactly as they appear
            5. Do not add information from outside the provided context
            6. If uncertain about any part of the answer, say so
            7. Use the same language and terminology as the document

            Context:
            {context}"""

            messages = [{"role": "system", "content": system_message}]
            if conversation_history:
                messages.extend(conversation_history[-2:])
            messages.append({"role": "user", "content": query})

            llm = ChatGroq(
                model=model,
                temperature=temperature,
                groq_api_key=Config.GROQ_API_KEY
            )

            response = llm.invoke(messages)
            return response.content[:2000] if len(response.content) > 2000 else response.content

        except Exception as e:
            error_str = str(e)
            if "413" in error_str and "tokens per minute (TPM)" in error_str:
                return json.dumps({
                    "type": "token_limit_error",
                    "message": (
                        "⚠️ The response requires more tokens than currently available.\n\n"
                        "Please try asking a more specific question or wait a minute before trying again."
                    )
                })
            logger.error(f"Error accessing vector store: {str(e)}")
            return "Error accessing knowledge base"

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return "Error processing request"