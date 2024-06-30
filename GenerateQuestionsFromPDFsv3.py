import fitz  # PyMuPDF
import os
import json
import re  # Import the regular expressions module
import ctypes  # Import ctypes to use the MessageBox function
import time  # Import time to implement the wait
import torch  # Import torch to check for CUDA availability
from lmqg import TransformersQG
from lmqg.exceptions import AnswerNotFoundError  # Import the specific exception

MAX_FILES_TO_PROCESS = 0  # Set to 0 to process all files

# Check for CUDA availability and set the default tensor type accordingly
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Using CUDA")
else:
    print("Using CPU")

# Initialize the model without specifying the device
model = TransformersQG(language="en")

def extract_text_from_pdfs(folder_path, max_files=MAX_FILES_TO_PROCESS):
    texts = []
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            if max_files != 0 and file_count >= max_files:  # Check if max_files is reached, unless max_files is 0
                break
            pdf_path = os.path.join(folder_path, filename)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                page_text = page.get_text()
                text += page_text
            texts.append(text)
            texts.append("\n\n")  # Separate texts by two newlines
            file_count += 1
    return texts

def preliminary_split_text(text, max_length=512, overlap=128):
    """
    Preliminarily splits the text into chunks that do not exceed max_length characters,
    using a sliding window approach with overlap for better context retention.
    """
    chunks = []
    current_start = 0
    text_length = len(text)

    while current_start < text_length:
        end_index = min(current_start + max_length, text_length)
        chunks.append(text[current_start:end_index])
        current_start += max_length - overlap

    return chunks

def split_text_into_smaller_chunks(text, tokenizer, max_length=512, overlap=128):
    """
    Splits a text into smaller chunks where each chunk has a token count within the model's limit.
    This function incorporates the use of preliminary_split_text to first split the text into
    larger chunks that do not exceed a certain character limit, ensuring that the text is manageable
    for tokenization without further need for splitting based on token count.
    """
    # Use preliminary_split_text to split the text into manageable chunks
    preliminary_chunks = preliminary_split_text(text, max_length=max_length, overlap=overlap)

    final_chunks = []
    for pre_chunk in preliminary_chunks:
        # Directly tokenize the pre-chunked text without further splitting
        tokens = tokenizer.tokenize(pre_chunk)
        token_count = len(tokens)
        
        if token_count > max_length:
            raise ValueError("Token count exceeds the maximum allowed length.")
        
        chunk_str = tokenizer.convert_tokens_to_string(tokens)
        final_chunks.append(chunk_str)

    return final_chunks
    

# Update the generate_qas function to use the updated split_text_into_smaller_chunks function
def generate_qas(texts, tokenizer, pdf_file_path):
    qas = []
    for text in texts:
        # Split each text into smaller parts to ensure they're within the token limit
        for part in split_text_into_smaller_chunks(text, tokenizer):
            try:
                # Process each part with the model
                qa_part = model.generate_qa([part])  # Assuming generate_qa can handle a list of texts
                qas.extend(qa_part)
            except AnswerNotFoundError as e:
                # Handle the error (e.g., log it, skip the part, etc.)
                print(f"Skipping a part due to AnswerNotFoundError: {e}")
    save_qas_to_json(qas, "qa_dataset.json", pdf_file_path)

def save_qas_to_json(qas, output_file, pdf_file_path):
    pdf_file_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
    
    modified_qas = []
    for qa_list in qas:  # Iterate through each list in qas
        for qa in qa_list:  # Iterate through each tuple in the list
            if isinstance(qa, tuple) and len(qa) == 2:  # Ensure qa is a tuple with exactly 2 elements
                question, answer = qa
            else:
                print(f"Invalid QA format: {qa}")
                continue  # Skip this iteration if the format is invalid
            
            modified_qa = {
                "Reference": pdf_file_name,
                "Question": "Question: " + question,
                "Answer": "Answer: " + answer
            }
            modified_qas.append(modified_qa)
    
    with open(output_file, "w") as f:
        json.dump(modified_qas, f, indent=4)


# In the main function or wherever you call generate_qas, pass the tokenizer from the model
def main(folder_path, output_file, max_files=MAX_FILES_TO_PROCESS):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            if max_files != 0 and len(texts) >= max_files:  # Check if max_files is reached, unless max_files is 0
                break
            pdf_path = os.path.join(folder_path, filename)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                page_text = page.get_text()
                text += page_text
            texts.append(text)
            tokenizer = model.tokenizer  # Assuming the model has a tokenizer attribute
            generate_qas([text], tokenizer, pdf_path)  # Pass the current PDF path
            print(f"Q&A dataset for {filename} saved to {output_file}")

    # Popup a message box before putting the computer to sleep
    MB_OKCANCEL = 0x01
    MB_ICONINFORMATION = 0x40
    result = ctypes.windll.user32.MessageBoxW(0, "The computer will go to sleep in 20 seconds. Press Cancel to abort.", "Sleep Timer", MB_OKCANCEL | MB_ICONINFORMATION)
    
    if result == 2:  # If the user pressed Cancel (IDCANCEL = 2)
        print("Sleep cancelled by the user.")
    else:
        print("No response or OK selected. Putting the computer to sleep.")
        time.sleep(20)  # Wait for 20 seconds before executing the sleep command
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

# specify the folder path and output file name
folder_path = "C:/Users/mclau/Downloads/DFARs-Processed"
output_file = "qa_dataset.json"
main(folder_path, output_file)
