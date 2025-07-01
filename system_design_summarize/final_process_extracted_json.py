import pickle
import json
import os
import requests
from tqdm import tqdm
from langchain.llms import Ollama
import time

# --- CONFIG ---
PICKLE_PATH = "system_design_summarize/ddia_langchain_documents.pkl"
STATE_PATH = "system_design_summarize/interview_intent_state.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
llm = Ollama(model=OLLAMA_MODEL)
FOCUS_AREAS = """
- Architecting, scaling, maintaining, or recovering critical systems
- Distributed systems principles (e.g. consensus, replication, coordination)
- Batch or streaming data pipeline design
- Fault tolerance, durability, availability, and recovery strategies
- Schema evolution (drift handling, enforcement, backward compatibility)
- Cost/performance trade-offs in design
- Monitoring, observability, or alerting best practices
"""

def load_documents():
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"processed": 0, "qa_pairs": {}}

def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

def ollama_query(prompt):
    response = llm(prompt)
    return response

def extract_qa_from_chunk(chunk):
    prompt = f"""
You are an expert interviewer for distributed systems and data engineering roles. Given the following technical content, extract:
1. The most likely interview question intent (one question, concise, relevant to these areas: {FOCUS_AREAS}).
2. Bullet points that would be expected in a strong answer.

Format your response as JSON:
{{
  "question": "...",
  "answer_points": ["...", "..."]
}}

Content:
\"\"\"{chunk}\"\"\"
"""
    response = ollama_query(prompt)
    # Try to extract JSON from response
    try:
        start = response.index('{')
        end = response.rindex('}') + 1
        return json.loads(response[start:end])
    except Exception:
        return None

def merge_qa_pairs(existing, new):
    # If question exists, merge answer points (no duplicates)
    for q, a in existing.items():
        if q.strip().lower() == new["question"].strip().lower():
            existing[q]["answer_points"] = list(set(existing[q]["answer_points"] + new["answer_points"]))
            return existing
    # Otherwise, add new question
    existing[new["question"]] = {"answer_points": new["answer_points"]}
    return existing

def main():
    documents = load_documents()
    state = load_state()
    start_idx = state["processed"]
    qa_pairs = state["qa_pairs"]

    for idx in tqdm(range(start_idx, len(documents)), desc="Processing chunks"):
        chunk = documents[idx].page_content
        qa = extract_qa_from_chunk(chunk)
        if qa and "question" in qa and "answer_points" in qa:
            qa_pairs = merge_qa_pairs(qa_pairs, qa)
        state["processed"] = idx + 1
        state["qa_pairs"] = qa_pairs
        save_state(state)
        time.sleep(20)  

    print(f"Processed {state['processed']} chunks. Total unique questions: {len(qa_pairs)}")
    # Optionally, save final QAs to a separate file
    with open("system_design_summarize/interview_questions.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)

if __name__ == "__main__":
    main()