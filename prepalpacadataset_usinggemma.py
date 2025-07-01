#Importing the necessary libraries
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.schema import Document
import json
from tqdm import tqdm
import pickle
import time

#Setting up the Ollama model
llm = Ollama(model="gemma3:4b")

#Defining functions
#Splitting docs
def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300,
    separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.split_documents(pages)
    return docs

# Local LLM invocation function
def local_llm(prompt: str, llm) -> str:
    """
    Sends a prompt to the LangChain Ollama LLM instance and returns the response.
    """
    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        print("Error invoking model:", e)
        return ""

#PRocess chunk function
def process_chunk_to_alpaca(doc: Document, llm) -> dict:
    # Extract metadata from the LangChain Document
    source_name = doc.metadata.get("source", "Unknown Name")

    # Inject metadata into prompt
    instruction_with_metadata = f"""
You are a business assistant analyzing raw business content from the following source:
SOURCE NAME: {source_name}

Your task is to extract the following from the provided transcript:
1. Frameworks (e.g., naming, advertising, validation models).
2. Bullet points for key ideas or steps.
3. Q&A (any implied or stated questions with answers).
4. Case Examples or stories.
5. Copywriting formulas (AIDA, PAS, etc.)
6. Classify this content into high-level topics: e.g., Naming, Ads, Psychology, Copywriting.
7. Convert suitable content into a step-by-step guide.

Return your output in clearly labeled sections, and only include sections with relevant content. Do not include a preamble.
""".strip()

    prompt = f"{instruction_with_metadata}\n\n{doc.page_content.strip()}"
    response = local_llm(prompt, llm)

    return {
        "instruction": instruction_with_metadata,
        "input": doc.page_content.strip(),
        "output": response,
        "metadata": doc.metadata
    }

#Saving processed file function

def load_jsonl_ids(filename):
    if not os.path.exists(filename):
        return set()
    with open(filename, "r") as f:
        return {json.loads(line).get("metadata", {}).get("source", "") + str(json.loads(line).get("metadata", {}).get("page", "")) for line in f}

def save_jsonl(filename, data):
    with open(filename, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def get_doc_id(doc: Document):
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page")

    if page is None:
        # Fallback to hashing part of the content if page is missing
        content_hash = str(abs(hash(doc.page_content[:50])))
        return f"{source}_hash_{content_hash}"

    return f"{source}_page_{page}"

def process_documents_with_retries(pages, llm):
    processed_ids = load_jsonl_ids(PROCESSED_FILE)
    failed_ids = load_jsonl_ids(FAILED_FILE)

    for doc in tqdm(pages, desc="Processing documents"):
        doc_id = get_doc_id(doc)

        if doc_id in processed_ids:
            continue

        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            try:
                alpaca_entry = process_chunk_to_alpaca(doc, llm)
                save_jsonl(PROCESSED_FILE, alpaca_entry)
                success = True
            except Exception as e:
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    error_entry = {
                        "error": str(e),
                        "metadata": doc.metadata,
                        "input": doc.page_content[:500]  # preview of failed input
                    }
                    save_jsonl(FAILED_FILE, error_entry)
        time.sleep(25)
        
        
#Setting up configs
PROCESSED_FILE = "marketing_books_alpaca_processed.jsonl"
FAILED_FILE = "marketing_books_alpaca_failed.jsonl"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# Update this path to your actual file location
pickle_file_path = '/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/marketing_pdfs.pkl'
# Load the data
with open(pickle_file_path, 'rb') as f:
    pickel_file = pickle.load(f)

all_docs = split_pages(pickel_file)
print(f"Total documents to process: {len(all_docs)}")

#Running our processing engine
process_documents_with_retries(all_docs, llm)