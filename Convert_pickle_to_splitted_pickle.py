#Importing the necessary libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

#Splitting docs
def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300,
    separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.split_documents(pages)
    return docs

def read_split_and_save_pickle(input_path, output_path):
    with open(input_path, 'rb') as f:
        pickel_file = pickle.load(f)

    all_docs = split_pages(pickel_file)
    print(f"Total documents to process: {len(all_docs)}")

    # Save each list of Documents separately
    with open(output_path, "wb") as f:
        pickle.dump(all_docs, f)
    
# Running our main function
pickle_file_input_path = '/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/marketing_pdfs.pkl'
pickle_file_output_path = '/Users/hitesh.modi/Desktop/Kinda Personal/LLM Finetuning Expert/marketing_pdfs_splitted.pkl'
read_split_and_save_pickle(pickle_file_input_path, pickle_file_output_path)