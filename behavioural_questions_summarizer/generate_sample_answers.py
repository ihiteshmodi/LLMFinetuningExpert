from langchain.llms import Ollama
import json
import time
from tqdm import tqdm

llm = Ollama(model="gemma3:4b")

INPUT_PATH = "/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/behavioural_questions_summarizer/unique_behavioral_questions.json"
OUTPUT_PATH = "/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/behavioural_questions_summarizer/sample_answers.json"

PROFILE = (
    "I am a senior data engineer in the biggest advertising agency in the globe. "
    "My tech key skills are: Python, SQL, PySpark, Spark Streaming, Kafka, AWS cloud, Apache suite of tools, and also Finetuning LLM's."
)

INSTRUCTION = (
    "Given the following behavioral interview question, generate a sample answer that demonstrates a great response from a senior data engineer in a top global advertising agency."
    "The answer should reflect expertise in Python, SQL, PySpark, Spark Streaming, Kafka, AWS, Apache tools, and LLM finetuning where relevant. "
    "Be specific, concise, and show both technical and communication skills."
    "No preamble"
)

def generate_sample_answer(question, llm):
    prompt = f"{INSTRUCTION}\n\nProfile: {PROFILE}\n\nQuestion: {question}\n\nSample Answer:"
    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        print("LLM error:", e)
        return ""

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        intent_to_question = json.load(f)

    answers = {}
    for intent, question in tqdm(intent_to_question.items(), desc="Generating sample answers"):
        answer = generate_sample_answer(question, llm)
        answers[question] = {
            "intent": intent,
            "sample_answer": answer
        }
        time.sleep(1)  # avoid rate limits

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    print(f"Saved sample answers to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()