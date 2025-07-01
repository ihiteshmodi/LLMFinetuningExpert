import json
from langchain.llms import Ollama
import time
from tqdm import tqdm

with open("/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/behavioural_questions_summarizer/behavioral_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

llm = Ollama(model="gemma3:4b")

def extract_meaning(question, llm):
    prompt = (
        "Given the following behavioral interview question, extract its core intent or meaning in one short sentence. "
        "Do not rephrase, just state the intent (e.g., 'assess teamwork', 'test leadership', 'evaluate conflict resolution').\n\n"
        f"Question: {question}\nMeaning:"
    )
    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        print("LLM error:", e)
        return ""

unique_meanings = []
unique_questions = []
meaning_to_question = {}

for q in tqdm(questions, desc="Deduplicating by meaning"):
    meaning = extract_meaning(q, llm)
    if not meaning:
        continue
    is_duplicate = False
    for saved_meaning in unique_meanings:
        # Use LLM to check if meanings are the same
        compare_prompt = (
            "Are the following two behavioral interview question meanings essentially the same? "
            "Answer only 'yes' or 'no'.\n"
            f"Meaning 1: {meaning}\nMeaning 2: {saved_meaning}\n"
        )
        try:
            response = llm.invoke(compare_prompt).strip().lower()
        except Exception as e:
            print("LLM error:", e)
            response = ""
        if response.startswith("yes"):
            is_duplicate = True
            break
        time.sleep(1)  # avoid rate limits
    if not is_duplicate:
        unique_meanings.append(meaning)
        unique_questions.append(q)
        meaning_to_question[meaning] = q

print(f"Unique questions by meaning: {len(unique_questions)}")

with open("/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/behavioural_questions_summarizer/unique_behavioral_questions.json", "w", encoding="utf-8") as f:
    json.dump(meaning_to_question, f, ensure_ascii=False, indent=2)

print("Saved unique questions and meanings to unique_behavioral_questions.json")