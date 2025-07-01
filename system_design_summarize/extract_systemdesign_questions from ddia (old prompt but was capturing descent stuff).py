import os
import pickle
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from tqdm import tqdm

# Load the LangChain Documents from pickle
with open("/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/ddia_langchain_documents.pkl", "rb") as f:
    documents = pickle.load(f)

# With a 128K token context window, we can use much larger chunks
# Let's use 12000 characters per chunk with 1000 overlap (adjust as needed for your use case)
splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=1000)
chunks = splitter.split_documents(documents)

# State management: track processed chunk indices
state_path = "/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/ddia_system_design_state.pkl"
output_path = "/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/ddia_system_design_chunks.json"

# If the state file does not exist, just start with an empty set (no error)
if os.path.exists(state_path):
    with open(state_path, "rb") as f:
        processed_indices = pickle.load(f)
else:
    processed_indices = set()

# If the output file does not exist, just start with an empty list (no error)
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        system_design_chunks = json.load(f)
else:
    system_design_chunks = []

llm = Ollama(model="gemma3:4b")

for idx, chunk in enumerate(tqdm(chunks, desc="Processing Chunks with LLM")):
    if idx in processed_indices:
        continue
    prompt = f"""
You are an expert data engineering interviewer. Given the following text chunk from a technical book, determine if it discusses any of the following system design topics relevant to data engineering:
- Ability to architect, scale, maintain, and recover critical systems
- Distributed systems
- Data pipeline design (batch/streaming)
- Fault tolerance and high availability
- Schema evolution (drift, enforcement, versioning)
- Cost/performance tradeoffs
- Monitoring/observability

If the chunk contains important information or advice about these topics, respond with 'YES' and a short summary of the relevant content. Otherwise, respond with 'NO'.

If you respond 'YES', also provide a concise sample answer (3-6 lines, not just 1 line, not the full chunk) that a senior data engineer could use in an interview, based on the content of the chunk.

Format your response as:
YES\nSummary: <summary>\nSampleAnswer: <sample answer>

Chunk:
"""
    prompt += chunk.page_content  # No restriction on length
    response = llm.invoke(prompt)
    if isinstance(response, dict):
        text = response.get('text', '').strip()
    else:
        text = str(response).strip()
    if text.startswith('YES'):
        # Try to extract summary and sample answer
        summary = ""
        sample_answer = ""
        for line in text.splitlines():
            if line.strip().lower().startswith('summary:'):
                summary = line.split(':', 1)[-1].strip()
            if line.strip().lower().startswith('sampleanswer:'):
                sample_answer = line.split(':', 1)[-1].strip()
        system_design_chunks.append({
            'chunk': chunk.page_content,
            'summary': summary,
            'sample_answer': sample_answer
        })
    processed_indices.add(idx)
    # Save state after each chunk
    with open(state_path, "wb") as f:
        pickle.dump(processed_indices, f)
    with open(output_path, "w") as f:
        json.dump(system_design_chunks, f, indent=2)

print(f"Done! Processed {len(processed_indices)} chunks. Extracted {len(system_design_chunks)} system design chunks.")