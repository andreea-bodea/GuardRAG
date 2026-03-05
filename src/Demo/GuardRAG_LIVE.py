# pip install -e .
# streamlit run Streamlit.py

import logging
import pandas as pd
import streamlit as st
import json
import asyncio
import plotly.express as px
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set logging level to ERROR to reduce noisy logs
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

from Demo.PDF_reader import convert_pdf_to_text
from Presidio.Presidio import analyze_text_with_presidio, delete_pii, label_pii, replace_pii
from Presidio.Presidio_helpers import annotate
from annotated_text import annotated_text
from Differential_privacy.DP import diff_privacy_diffractor, diff_privacy_dp_prompt, diff_privacy_dpmlm
from RAG.Response_evaluation import calculate_rouge1, calculate_rougeL, calculate_bleu, calculate_cosine_similarity, calculate_perplexity, calculate_privacy_llm_judge, extract_llm_score
from RAG.Local_LlamaIndex import get_offline_RAG_response

try:
    st.set_page_config(
        page_title="GuardRAG",
        layout="wide", #"centered" 
        initial_sidebar_state="collapsed" #"expanded",
    )
except Exception as e:
    pass 

st.title("GuardRAG LIVE 🔍")
st.subheader("Protecting private data in retrieval-augmented generation systems.")

# About section with information about the app
with st.expander("**‼️ About GuardRAG**", expanded=False):
    st.markdown("""
    ### About GuardRAG
    
    GuardRAG is a real-time Retrieval-Augmented Generation (RAG) system, 
    that demonstrates how to protect sensitive information in RAG systems using 
    various anonymization techniques and differential privacy-based methods  
    while preserving the utility of generated responses.
    
    #### Privacy-preserving Methods
        
    **A. Anonymization**:
    
    **PII Deletion**  
    - Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
    - Completely removes all detected entities   
    - ✅ **Advantages**: Maximizes privacy by eliminating sensitive content entirely  
    - ⚠️ **Disadvantages**: Can reduce coherence and remove useful context

    **PII Labeling**  
    - Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
    - Replaces entities with generic labels such as `<PERSON>`, `<LOCATION>`, `<DATE_TIME>`  
    - ✅ **Advantages**: Maintains sentence structure and grammar; reversible if mappings are saved  
    - ⚠️ **Disadvantages**: Generic labels may reduce readability and interpretability

    **PII Replacement with Synthetic Data**  
    - Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
    - Replaces detected entities with placeholders (e.g., `<PERSON>`, `<DATE>`)  
    - Uses OpenAI's `gpt-3.5-turbo-instruct` to generate realistic, synthetic replacements  
    - ✅ **Advantages**: Retains fluency and realism; preserves structure while protecting privacy  
    - ⚠️ **Disadvantages**: More computationally expensive and reliant on the generative model's quality
                
    **B. Paraphrasing based on Differential Privacy (DP)**: 

    **1-Diffractor**
    - Tokenizes the input text 
    - Classifies the tokens and ignores stopwords (e.g. "the", "and") and punctuation
    - For each remaining token, creates embeddings from pre-trained models like GloVe or Word2Vec
    - Uses FAISS to finds the top N semantically similar tokens (neighbors)
    - Instead of choosing the closest token, a **geometric DP mechanism** selects one at random, with higher chance for similar words
    - Replaces the original token with the chosen 
    - 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 1 or 2 or 3.
    - ✅ **Advantages**: Protects against re-identification by making words fuzzy, not just redacted; Preserves general meaning better than deletion
    - ⚠️ **Disadvantages**: Depending on the chosen epsilon, it may change the meaning of the sentence; too much randomness may hurt clarity of the sentence
    
    **DP-Prompt**: 
    - Tokenizes the input text using a transformer tokenizer
    - Formats the input like into "Document: [original text]. Paraphrase of the document: "
    - Clipps the model's prediction scores (logits) to limit their range (Logit Clipping)
    - Calculates the sensitivity (how much influence one token can have) using the clipped range
    - Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
    - Generates a paraphrased version of the text through the transformer model using **temperature-controlled sampling**
    - 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 150 or 200 or 250. 
    - ✅ **Advantages**: Produces full, fluent paraphrases instead of word-by-word changes; Works well at sentence-level.
    - ⚠️ **Disadvantages**: Too low ε results in vague, truncated, or confusing outputs; Not great for **short texts** or word-level use.

    **DP-MLM (Differentially Private Masked Language Model)**
    - Masks each word in the sentence one at a time
    - Pairs the masked sentence with the original version to give context
    - Uses a masked language model to predicts the hidden word
    - Clips the output logits to limit sensitivity
    - Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
    - Samples the prediction using the temperature 
    - Replaces the masked word with the predicted word
    - 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 50 or 75 or 100.
    - ✅ **Advantages**: Preserves sentence structure and meaning better than simple word substitution; High flexibility for granular anonymization.
    - ⚠️ **Disadvantages**: Too short inputs result in low-quality predictions; Outputs can lose fluency if context is poor or ε is very low.                
    
    #### Evaluation Methods
        
    **Utility Metrics**:
    - **Rouge-1**: Measures unigram overlap (recall-focused)
    - **Rouge-L**: Evaluates longest common subsequence similarity
    - **BLEU**: Calculates n-gram precision compared to a reference
    - **Cosine Similarity**: Measures semantic closeness between response embeddings
    - **Perplexity**: Evaluates language fluency (lower values are better)
    
    **Privacy Metric**:
    - **LLM-as-a-Judge**: Uses an LLM to determine the percentage of privacy leakage in the generated response
    
    The evaluation process helps determine the balance between maintaining response utility while protecting sensitive information.
    """)

st_logger = logging.getLogger('streamlit')
st_logger.setLevel(logging.INFO)

# Initialize session state for persistent storage
if 'text_with_pii' not in st.session_state:
    st.session_state.text_with_pii = None
if 'text_anonymized' not in st.session_state:
    st.session_state.text_anonymized = None
if 'selected_anonymization' not in st.session_state:
    st.session_state.selected_anonymization = None

uploaded_file = None
text_with_pii = None
text_anonymized = None
selected_anonymization = None
selected_epsilon = None
response_original = None
response_anonymized = None

### DATA SELECTION ###

st.markdown("---")
st.subheader("📝 ANONYMIZATION / PARAPHRASING : upload a file or type in a text and choose the privacy-preserving method")

# Initialize lang_code with default value
lang_code = "en"

col1, col2, col3 = st.columns([1, 1, 1])

# INPUT - COLUMN 1: UPLOADED FILE
uploaded_file = col1.file_uploader("Upload a file:", type=["pdf", "txt"])

# Add example text options to column 1
with col1.expander("📋 Load example text"):
    example_texts = {
        "Personal Email": """From: john.smith@gmail.com
To: sarah.johnson@company.co.uk
Date: May 15, 2023
Subject: Meeting next week

Hi Sarah,

I hope this email finds you well. I wanted to confirm our meeting next Thursday, May 25, at 2:00 PM at our office (123 Main Street, London).

Please bring your ID and the contract documents we discussed. If you need to reschedule, please call me at +44 7700 123456.

My assistant, Tom Wilson, will be joining us. He can be reached at tom.wilson@example.com if you need anything beforehand.

Bank details for expense reimbursement:
Account Name: John Smith
Account Number: 12345678
Sort Code: 10-20-30

Best regards,
John Smith
Marketing Director
ID: GB9834567""",

        "Simple Contract": """RENTAL AGREEMENT

Date: June 1, 2023

BETWEEN:
Jane Doe (Landlord)
Address: 45 Park Avenue, New York, NY 10022
Phone: 212-555-1234
Email: jane.doe@email.com

AND:
Robert Johnson (Tenant)
Phone: 917-555-6789
Email: robert.johnson@email.com

PROPERTY:
123 Maple Street, Apt 4B
New York, NY 10001

TERMS:
1. Rental period: June 15, 2023 to June 14, 2024
2. Monthly rent: $2,500 due on the 1st of each month
3. Security deposit: $3,000

Payment details:
Bank: First National Bank
Account: Jane Doe
Account Number: 987654321
"""
    }
    
    if st.button("Load Email Example"):
        st.session_state['example_text'] = example_texts["Personal Email"]
        st.rerun()
        
    if st.button("Load Contract Example"):
        st.session_state['example_text'] = example_texts["Simple Contract"]
        st.rerun()

if uploaded_file is not None:
    with st.spinner('Reading text...'):
        try:
            file_extension = uploaded_file.name.split('.')[-1]
            file_bytes = uploaded_file.read()
            st_logger.info(f"File uploaded: {uploaded_file.name}, size: {len(file_bytes)} bytes, type: {file_extension}")
            
            if file_extension == "pdf":
                st_logger.info("Processing PDF file...")
                # Use current lang_code (will be default "en" or previously selected language)
                text_with_pii = asyncio.run(convert_pdf_to_text(file_bytes, language=lang_code))
                st_logger.info(f"PDF text extracted, length: {len(text_with_pii) if text_with_pii else 0} characters")
                if not text_with_pii or len(text_with_pii.strip()) == 0:
                    st.warning("The PDF couldn't be read or contains no text. Please try another file.")
            elif file_extension == "txt":
                text_with_pii = file_bytes.decode("utf-8")
                st_logger.info(f"TXT file read, length: {len(text_with_pii)} characters")
            
            # Reset file position for future reads
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st_logger.error(f"Error reading file: {str(e)}", exc_info=True)
            text_with_pii = None

# INPUT - COLUMN 2: TYPED TEXT
with col2:
    # Initialize the example text state if it doesn't exist
    if 'example_text' not in st.session_state:
        st.session_state['example_text'] = ""
    
    # Use the example text if it exists in session state
    initial_text = st.session_state['example_text'] if st.session_state['example_text'] else ""
    typed_text = st.text_area("Or type in a text:", value=initial_text, height=200)
    send_button_clicked = st.button("Send")
    
    # Only use typed text if it's not empty and there's no uploaded file text
    if typed_text and (not text_with_pii or len(text_with_pii.strip()) == 0):
        text_with_pii = typed_text
        st_logger.info(f"Using typed text, length: {len(text_with_pii)} characters")

# INPUT - COLUMN 3 

anonymization_types = [
    "PII Deletion",
    "PII Labeling",
    "PII Replacement with Synthetic Data",
    "Diffractor", 
    "DP-Prompt", 
    "DP-MLM"
]
anonymization_type_map = {
    "PII Deletion": "text_pii_deleted",
    "PII Labeling": "text_pii_labeled",
    "PII Replacement with Synthetic Data": "text_pii_synthetic",
    "Diffractor": "text_pii_dp_diffractor",
    "DP-Prompt": "text_pii_dp_dp_prompt",
    "DP-MLM": "text_pii_dp_dpmlm"
}

mitigations_explanation = """
**PII Deletion**  
- Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
- Completely removes all detected entities   
- ✅ **Advantages**: Maximizes privacy by eliminating sensitive content entirely  
- ⚠️ **Disadvantages**: Can reduce coherence and remove useful context

**PII Labeling**  
- Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
- Replaces entities with generic labels such as `<PERSON>`, `<LOCATION>`, `<DATE_TIME>`  
- ✅ **Advantages**: Maintains sentence structure and grammar; reversible if mappings are saved  
- ⚠️ **Disadvantages**: Generic labels may reduce readability and interpretability

**PII Replacement with Synthetic Data**  
- Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
- Replaces detected entities with placeholders (e.g., `<PERSON>`, `<DATE>`)  
- Uses OpenAI's `gpt-3.5-turbo-instruct` to generate realistic, synthetic replacements  
- ✅ **Advantages**: Retains fluency and realism; preserves structure while protecting privacy  
- ⚠️ **Disadvantages**: More computationally expensive and reliant on the generative model's quality

**1-Diffractor**
- Tokenizes the input text 
- Classifies the tokens and ignores stopwords (e.g. "the", "and") and punctuation
- For each remaining token, creates embeddings from pre-trained models like GloVe or Word2Vec
- Uses FAISS to finds the top N semantically similar tokens (neighbors)
- Instead of choosing the closest token, a **geometric DP mechanism** selects one at random, with higher chance for similar words
- Replaces the original token with the chosen 
- 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 1 or 2 or 3.
- ✅ **Advantages**: Protects against re-identification by making words fuzzy, not just redacted; Preserves general meaning better than deletion
- ⚠️ **Disadvantages**: Depending on the chosen epsilon, it may change the meaning of the sentence; too much randomness may hurt clarity of the sentence
 
**DP-Prompt**: 
- Tokenizes the input text using a transformer tokenizer
- Formats the input like into "Document: [original text]. Paraphrase of the document: "
- Clipps the model's prediction scores (logits) to limit their range (Logit Clipping)
- Calculates the sensitivity (how much influence one token can have) using the clipped range
- Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
- Generates a paraphrased version of the text through the transformer model using **temperature-controlled sampling**
- 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 150 or 200 or 250.
- ✅ **Advantages**: Produces full, fluent paraphrases instead of word-by-word changes; Works well at sentence-level.
- ⚠️ **Disadvantages**: Too low ε results in vague, truncated, or confusing outputs; Not great for **short texts** or word-level use.

**DP-MLM (Differentially Private Masked Language Model)**
- Masks each word in the sentence one at a time
- Pairs the masked sentence with the original version to give context
- Uses a masked language model to predicts the hidden word
- Clips the output logits to limit sensitivity
- Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
- Samples the prediction using the temperature 
- Replaces the masked word with the predicted word
- 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 50 or 75 or 100.
- ✅ **Advantages**: Preserves sentence structure and meaning better than simple word substitution; High flexibility for granular anonymization.
- ⚠️ **Disadvantages**: Too short inputs result in low-quality predictions; Outputs can lose fluency if context is poor or ε is very low.
"""
selected_anonymization = col3.selectbox(
    "Select privacy-preserving method:", 
    anonymization_types, 
    help=mitigations_explanation,
    label_visibility="visible",
    index=0
)

# Recommended epsilon values based on method
epsilon_recommendations = {
    "Diffractor": {"en": (1, 10)},
    "DP-Prompt": {"en": (0, 300)},
    "DP-MLM": {"en": (0, 300)}
}

# Customize the epsilon slider based on selected method
if selected_anonymization in ["Diffractor", "DP-Prompt", "DP-MLM"]:
    min_value, max_value = epsilon_recommendations.get(selected_anonymization, {}).get(lang_code, (0, 300))
    default_value = (min_value + max_value) // 2
    
    epsilon_help = f"Recommended epsilon range for {selected_anonymization}: {min_value}-{max_value}. Lower values provide more privacy but less utility."
    selected_epsilon = col3.slider(
        label="Epsilon value",
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        help=epsilon_help,
    )
else:
    # Hide epsilon slider for non-DP methods
    selected_epsilon = 0

### TEXT ANONYMIZATION / PARAPHRASING ###

if send_button_clicked:

    if text_with_pii is None or len(text_with_pii.strip()) == 0: 
        st.error("Please upload a file or type in some text.")
    elif selected_anonymization is None: 
        st.error("Please select an anonymization method.")
    else:
        col1, col2 = st.columns([1, 1])

        col1.subheader(f"📄 Original text")
        col1.text_area(
            label="Original Text:",
            value=text_with_pii,
            height=400,
            label_visibility="hidden" # visible, hidden, collapsed
        )
        # Store in session state
        st.session_state.text_with_pii = text_with_pii
        st.session_state.selected_anonymization = selected_anonymization

        if selected_anonymization in ["Diffractor", "DP-Prompt", "DP-MLM"] and selected_epsilon == 0:
            st.error("Please select a non-zero epsilon value.")
        else:
            with st.spinner('Anonymizing text...'):
                try:
                    # Add progress placeholder for DP methods which can be slow
                    if selected_anonymization in ["DP-Prompt", "DP-MLM", "Diffractor"]:
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        progress_text.text("Initializing models...")
                        
                        # For DP methods
                        if selected_anonymization == "Diffractor":
                            progress_text.text("Applying differential privacy with Diffractor...")
                            progress_bar.progress(30)
                            text_anonymized = diff_privacy_diffractor(text_with_pii, selected_epsilon, language=lang_code)
                            progress_bar.progress(100)
                            progress_text.text("Anonymization complete!")
                        elif selected_anonymization == "DP-Prompt":
                            progress_text.text("Applying DP-Prompt (this may take a while)...")
                            progress_bar.progress(30)
                            text_anonymized = diff_privacy_dp_prompt(text_with_pii, selected_epsilon, language=lang_code)
                            progress_bar.progress(100)
                            progress_text.text("Anonymization complete!")
                        elif selected_anonymization == "DP-MLM":
                            progress_text.text("Applying DP-MLM (this may take a while)...")
                            progress_bar.progress(30)
                            text_anonymized = diff_privacy_dpmlm(text_with_pii, selected_epsilon, language=lang_code)
                            progress_bar.progress(100)
                            progress_text.text("Anonymization complete!")
                    else:
                        # For non-DP methods
                        if selected_anonymization == "PII Deletion":
                            text_anonymized = delete_pii(text_with_pii, language=lang_code)
                        elif selected_anonymization == "PII Labeling":
                            text_anonymized = label_pii(text_with_pii, language=lang_code)
                        elif selected_anonymization == "PII Replacement with Synthetic Data":
                            text_anonymized = replace_pii(text_with_pii, language=lang_code)
                    
                    # Check if anonymization produced a result
                    if not text_anonymized or len(text_anonymized.strip()) == 0:
                        st.error("Anonymization process resulted in empty text. Please check your input text.")
                        text_anonymized = "Anonymization process did not produce a result."
                    
                    # Store anonymized text in session state
                    st.session_state.text_anonymized = text_anonymized
            
                    col2.subheader(f"🔐 Text after {selected_anonymization}")
                    col2.text_area(
                        label="Anonymized Text:",
                        value=text_anonymized,
                        height=400,
                        label_visibility="hidden" # visible, hidden, collapsed
                    )
                except Exception as e:
                    st.error(f"Error during anonymization: {str(e)}")
                    st_logger.error(f"Error during anonymization: {str(e)}", exc_info=True)
                    
                    # Display a more user-friendly error message based on the method
                    if selected_anonymization == "PII Replacement with Synthetic Data" and "OPENAI_API_KEY" in str(e):
                        st.warning("To use the PII Replacement with Synthetic Data method, you need to configure your OpenAI API key in the .env file.")
                    elif "CUDA" in str(e) or "GPU" in str(e):
                        st.warning("This method requires GPU resources that might not be available. Try using a different anonymization method.")
                    
                    # Reset anonymized text to prevent using incomplete results
                    st.session_state.text_anonymized = None

    if selected_anonymization in ["PII Deletion", "PII Labeling", "PII Replacement with Synthetic Data"]:
        try:
            st_analyze_results = analyze_text_with_presidio(text_with_pii, language=lang_code)
            results_as_dicts = [result.to_dict() for result in st_analyze_results]
            results_json = json.dumps(results_as_dicts, indent=2)

            with st.expander("Personally Identifiable Information (PII) detected in the original text", expanded=False):
                annotated_tokens = annotate(text=text_with_pii, analyze_results=st_analyze_results)
                annotated_text(*annotated_tokens)

            with st.expander("Detailed Findings", expanded=False):
                if results_json and st_analyze_results:
                    if len(st_analyze_results) > 0:
                        df_text = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
                        df_text["Text"] = [text_with_pii[res.start:res.end] for res in st_analyze_results]
                        df_subset = df_text[["entity_type", "Text", "start", "end", "score"]].rename(
                            {
                                "entity_type": "Entity type",
                                "start": "Start",
                                "end": "End",
                                "score": "Confidence",
                            },
                            axis=1,
                        )
                        st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
                    else:
                        st.text("No Personally identifiable information (PII) found")
                else:
                    st.text("No Personally identifiable information (PII) found")
        except Exception as e:
            st.error(f"Error displaying PII information: {str(e)}")
            st_logger.error(f"Error in PII display: {str(e)}", exc_info=True)
            st.warning("Continuing with anonymization, but PII detection results cannot be displayed.")
            
            # Log a warning but allow the app to continue
            st_logger.warning("PII detection issue but continuing with anonymization")

# Always display text boxes if they exist in session state (and not already shown by send_button)
if st.session_state.text_with_pii is not None and st.session_state.text_anonymized is not None and not send_button_clicked:
    col1, col2 = st.columns([1, 1])
    
    col1.subheader(f"📄 Original text")
    col1.text_area(
        label="Original Text:",
        value=st.session_state.text_with_pii,
        height=300,
        label_visibility="hidden",
        disabled=True
    )
    
    col2.subheader(f"🔐 Text after {st.session_state.selected_anonymization}")
    col2.text_area(
        label="Anonymized Text:",
        value=st.session_state.text_anonymized,
        height=300,
        label_visibility="hidden",
        disabled=True
    )

### RAG ###

st.markdown("---")
st.subheader("📝 RAG: ask a question about the text and compare the response based on the original text with the one based on the anonymized text")
st_logger.info("Waiting for user input...")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    st_logger.info("Get Answer button clicked.")

    # Check if text has been processed
    if st.session_state.text_with_pii is None:
        st.error("Please provide some text first and click 'Send' to anonymize it.")
    elif st.session_state.text_anonymized is None:
        st.error("Please anonymize your text by clicking 'Send' first.")
    elif not question:
        st.error("Please enter a question.")
    else:
        with st.spinner('Generating responses...'):
            response_original_obj = get_offline_RAG_response(question, st.session_state.text_with_pii)
            response_anonymized_obj = get_offline_RAG_response(question, st.session_state.text_anonymized)
            
            # Extract text content from response objects
            response_original = str(response_original_obj)
            response_anonymized = str(response_anonymized_obj)

        col1, col2 = st.columns([1, 1])
        col1.subheader(f"📄 Response based on the original text")
        col1.text_area(
            label="Response on the original text", 
            value=response_original, 
            height=200, 
            key="response_original",
            label_visibility="hidden" # visible, hidden, collapsed
        )

        col2.subheader(f"🔐 Response based on the text after {st.session_state.selected_anonymization}")
        col2.text_area(
            label="Response on the anonymized text", 
            value=response_anonymized, 
            height=200, 
            key="response_anonymized",
            label_visibility="hidden" # visible, hidden, collapsed
        )

### Evaluation ###

        rouge1_score = round(calculate_rouge1(response_original, response_anonymized), 2)
        rougeL_score = round(calculate_rougeL(response_original, response_anonymized), 2)
        bleu_score = round(calculate_bleu(response_original, response_anonymized), 2)
        cosine_similarity_score = round(calculate_cosine_similarity(response_original, response_anonymized), 2)
        perplexity_score = round(calculate_perplexity(response_anonymized), 2)
        llm_score = extract_llm_score(calculate_privacy_llm_judge(st.session_state.text_with_pii, response_anonymized))

        evaluation_df = pd.DataFrame({
            "Metric": [
                "Rouge-1", "Rouge-L", "BLEU", "Cosine Similarity", "Perplexity", "LLM-as-a-Judge"
            ],
            "Score": [
                rouge1_score, rougeL_score, bleu_score,
                cosine_similarity_score, perplexity_score, llm_score
            ],
            "Explanation": [
                "Overlap of unigrams (recall-focused)",
                "Longest common subsequence (sequence similarity)",
                "N-gram precision of generated vs reference",
                "Semantic closeness of embeddings",
                "How predictable the text is (lower = better)",
                "LLM-based judgment on percentage of privacy leakage"
            ]
        })

        # Normalize metrics where higher = better
        normalized_scores = []

        for metric, score in zip(evaluation_df["Metric"], evaluation_df["Score"]):
            if metric == "Perplexity":
                normalized = 1 / score if score > 0 else 0
            elif metric == "LLM-as-a-Judge":
                normalized = 1 - (score / 100)
            else:
                normalized = score  # assumed in 0-1 range
            normalized_scores.append(normalized)

        # Scale to [0, 1] range based on max
        evaluation_df["Normalized"] = normalized_scores
        evaluation_df["Normalized"] /= max(normalized_scores)

        col1.subheader(f"📊 Evaluation of the response based on the text after {st.session_state.selected_anonymization}")
        col1.dataframe(evaluation_df)

        bar_fig = px.bar(
            evaluation_df,
            x="Metric",
            y="Normalized",
            hover_data=["Explanation", "Score"],
            color="Metric",
            title="",
            labels={"Normalized": "Normalized Score (0–1)"},
        )
        col2.subheader(f"📊 Barchart for Normalized Evaluation Metrics")
        col2.plotly_chart(bar_fig, use_container_width=True)
