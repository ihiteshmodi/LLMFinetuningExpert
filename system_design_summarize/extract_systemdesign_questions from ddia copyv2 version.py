import os
import pickle
import json
import time
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from tqdm import tqdm

# --- Helper functions for filtering ---
def is_index_or_glossary(chunk_text):
    lines = chunk_text.splitlines()
    # Heuristic: if >60% of lines are very short or contain mostly numbers/punctuation, it's likely an index/glossary
    short_lines = sum(1 for l in lines if len(l.strip()) < 40)
    punct_num_lines = sum(1 for l in lines if re.fullmatch(r'[\W\d\s]+', l.strip()))
    digit_ratio = sum(c.isdigit() for c in chunk_text) / max(1, len(chunk_text))
    # If more than 60% lines are short or punct/num, or digit ratio is high, skip
    if len(lines) > 5 and (short_lines / len(lines) > 0.6 or punct_num_lines / len(lines) > 0.4 or digit_ratio > 0.15):
        return True
    # Heuristic: if chunk contains 'index', 'table of contents', 'glossary', skip
    lowered = chunk_text.lower()
    if any(word in lowered for word in ['index', 'table of contents', 'glossary']):
        return True
    return False

# --- Section-based chunking ---
def section_chunk_documents(documents, min_section_len=1000, max_section_len=12000):
    # Try to chunk by headings (lines with all caps or starting with numbers/chapters)
    section_chunks = []
    buffer = ""
    for doc in documents:
        lines = doc.page_content.splitlines()
        for line in lines:
            # Heading detection: all caps, or starts with 'chapter', or is a number+dot
            if (re.match(r'^[A-Z\s\-:]{8,}$', line.strip()) or
                re.match(r'^(chapter|section|part)\b', line.strip().lower()) or
                re.match(r'^\d+\.\s', line.strip())):
                if len(buffer) > min_section_len:
                    section_chunks.append(buffer)
                    buffer = ""
            buffer += line + "\n"
            if len(buffer) > max_section_len:
                section_chunks.append(buffer)
                buffer = ""
        if len(buffer) > min_section_len:
            section_chunks.append(buffer)
            buffer = ""
    return section_chunks

# Load the LangChain Documents from pickle
with open("/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/ddia_langchain_documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Use section-based chunking for more meaningful context
chunks = section_chunk_documents(documents)

# State management: track processed chunk indices
state_path = "/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/ddia_system_design_state.pkl"
output_path = "/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/ddia_system_design_chunks.json"

if os.path.exists(state_path):
    with open(state_path, "rb") as f:
        processed_indices = pickle.load(f)
else:
    processed_indices = set()

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        system_design_chunks = json.load(f)
else:
    system_design_chunks = []

llm = Ollama(model="gemma3:4b")

for idx, chunk_text in enumerate(tqdm(chunks, desc="Processing Chunks with LLM")):
    if idx in processed_indices:
        continue
    # Pre-filter: skip index/glossary/TOC chunks
    if is_index_or_glossary(chunk_text):
        processed_indices.add(idx)
        continue
    prompt = f"""
You are an expert data engineering interviewer and system designer.\n\nYou are analyzing a chunk of text from a technical book to extract **only deep system design insights** that are useful for answering senior data engineering interview questions.\n\nIgnore tool-specific details, vendor technologies, and implementation code. Focus purely on principles, architecture patterns, trade-offs, and reliability strategies.\n\nDetermine whether the chunk discusses **any** of the following key system design themes:\n\n- Architecting, scaling, maintaining, or recovering critical systems\n- Distributed systems principles (e.g. consensus, replication, coordination)\n- Batch or streaming data pipeline design\n- Fault tolerance, durability, availability, and recovery strategies\n- Schema evolution (drift handling, enforcement, backward compatibility)\n- Cost/performance trade-offs in design\n- Monitoring, observability, or alerting best practices\n\nIf the chunk contains useful insights about **any** of these topics, respond with:\n\nYES  \nSummary: <1–2 line summary>  \nSampleAnswer: <3–6 line response that a senior data engineer might give in an interview using this idea>  \nPotentialInterviewQuestion: <Optional — generate a realistic interview question based on the chunk>\n\nIf the chunk is just a list of terms, an index, a glossary, or contains no relevant design-level content, respond with:\n\nNO\n\nChunk:\n""" + chunk_text
    response = llm.invoke(prompt)
    time.sleep(2)
    if isinstance(response, dict):
        text = response.get('text', '').strip()
    else:
        text = str(response).strip()
    if text.startswith('YES'):
        summary = ""
        sample_answer = ""
        interview_question = ""
        for line in text.splitlines():
            if line.strip().lower().startswith('summary:'):
                summary = line.split(':', 1)[-1].strip()
            if line.strip().lower().startswith('sampleanswer:'):
                sample_answer = line.split(':', 1)[-1].strip()
            if line.strip().lower().startswith('potentialinterviewquestion:'):
                interview_question = line.split(':', 1)[-1].strip()
        system_design_chunks.append({
            'chunk': chunk_text,
            'summary': summary,
            'sample_answer': sample_answer,
            'potential_interview_question': interview_question
        })
    processed_indices.add(idx)
    with open(state_path, "wb") as f:
        pickle.dump(processed_indices, f)
    with open(output_path, "w") as f:
        json.dump(system_design_chunks, f, indent=2)

print(f"Done! Processed {len(processed_indices)} chunks. Extracted {len(system_design_chunks)} system design chunks.")