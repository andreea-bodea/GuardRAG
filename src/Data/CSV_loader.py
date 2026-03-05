import logging
import csv
import hashlib
import nltk
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.Database_management import retrieve_record_by_hash
from Data_loader import load_data_all, load_data_text_with_pii, load_data_presidio, load_data_diffractor, load_data_dp_prompt, load_data_dpmlm

st_logger = logging.getLogger('CSV_loader')
st_logger.setLevel(logging.INFO)

# Add console handler if not already present
if not st_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    st_logger.addHandler(handler)
        
def load(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, type):
    # If only saving original text, check if it exists first
    if type == 'text_with_pii':
        existing_record = retrieve_record_by_hash(table_name, file_hash)
        if existing_record is None or existing_record.get('text_with_pii') is None:
            load_data_text_with_pii(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
        else:
            st_logger.info(f"Original text already exists for {file_name}, skipping insert")
        return retrieve_record_by_hash(table_name, file_hash)
    
    # For all anonymization methods, ensure original text is saved first
    existing_record = retrieve_record_by_hash(table_name, file_hash)
    if existing_record is None or existing_record.get('text_with_pii') is None:
        st_logger.info(f"Original text not found for {file_name}. Saving original text first...")
        load_data_text_with_pii(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    
    # Now process with anonymization methods
    if type == 'all':
        load_data_all(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    elif type == 'presidio':
        load_data_presidio(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    elif type == 'diffractor':
        load_data_diffractor(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    elif type == 'dp_prompt':
        load_data_dp_prompt(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    elif type == 'dpmlm':
        load_data_dpmlm(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    else:
        st_logger.error(f"Unknown anonymization type: {type}")
        return None
        
    database_file = retrieve_record_by_hash(table_name, file_hash)
    return database_file
             
def load_enron(type):
    file_path = "./Enron_preprocessed.csv"
    index_name = "enron2"
    table_name = "enron_text3"

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the first row (header)
        for row_number, row in enumerate(csv_reader, start=1):
            # if row_number == 61: continue  
            if row_number < 100: continue
            if row_number > 100: break
            text_with_pii = row[0]
            file_name = f"Enron_{row_number}"
            file_bytes = text_with_pii.encode("utf-8") 
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            st_logger.info(f"File hash: {file_hash}")
            load(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, type=type)

def load_bbc(type):
    file_path = "./BBC_preprocessed.csv"
    index_name = "bbc2"
    table_name = "bbc_text2"

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the first row (header)
        for row_number, row in enumerate(csv_reader, start=1):
            text_with_pii = row[0]
            file_name = f"BBC_{row_number}"
            file_bytes = text_with_pii.encode("utf-8")
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            st_logger.info(f"Started loading: {file_name}")
            load(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, type=type)

def load_tab(type):
    file_path = "./TAB_first_1000_echr_train.csv"
    index_name = "tab"
    table_name = "tab_text2"

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row_number, row in enumerate(csv_reader, start=1):
            if row_number <= 2: continue
            if row_number > 200: break
            text_with_pii = row[0]
            file_name = f"TAB_{row_number}"
            file_bytes = text_with_pii.encode("utf-8")
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            st_logger.info(f"Started loading: {file_name}")
            load(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, type=type)

if __name__ == "__main__":
    # type: 'text_with_pii', 'presidio', 'diffractor', 'dp_prompt', 'dpmlm', 'all'
    # nltk.download('punkt')
    # nltk.download('punkt_tab')
    #load_enron(type='dpmlm')
    #load_bbc(type='dp_prompt')
    #load_tab(type='all')
    load_tab(type='presidio')