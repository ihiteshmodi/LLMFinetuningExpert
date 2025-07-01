from langchain_community.document_loaders.pdf import PyPDFLoader
import pickle


# Path to the PDF file
pdf_path = "/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/Designing Data-Intensive Applications The Big Ideas Behind Reliable, Scalable, and Maintainable Systems by Martin Kleppmann (z-lib.org).pdf"

# Load the PDF as LangChain Documents
loader = PyPDFLoader(pdf_path)
documents = list(loader.lazy_load())

# Save the LangChain Documents object locally as a pickle file
with open("/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/system_design_summarize/ddia_langchain_documents.pkl", "wb") as f:
    pickle.dump(documents, f)