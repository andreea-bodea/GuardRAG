# pip install -e .
# streamlit run Streamlit.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import streamlit as st
import pandas as pd
import json
from natsort import natsorted
import plotly.express as px
from presidio_analyzer import RecognizerResult
from annotated_text import annotated_text
from Presidio.Presidio_helpers import (
    annotate,
)    

def extract_llm_score(entry):
    """
    Extracts the privacy_leakage_score from an entry.
    If the score is None, it checks the explanation for a fallback value.
    """
    judge = entry.get("privacy_llm_judge", {})
    score = judge.get("privacy_leakage_score")
    
    # If score is available (and not None), return it as a float.
    if score is not None:
        return float(score)
    
    # If score is None, check the explanation text.
    explanation = judge.get("explanation", "")
    if "failed to parse json" in explanation.lower() and "privacy_leakage_score" in explanation:
        # Look for a numeric value after "privacy_leakage_score"
        match = re.search(r'"privacy_leakage_score"\s*:\s*([\d\.]+)', explanation)
        if match:
            return float(match.group(1))
    return 0

st.set_page_config(
    page_title="GuardRAG",
    layout="wide", #"centered" 
    initial_sidebar_state="collapsed" #"expanded",
)

st.title("GuardRAG 🔍")
st.subheader("Protecting private data in retrieval-augmented generation systems.")

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

### DATA SELECTION ###

col1, col2, col3 = st.columns([1, 1, 1])

datasets = ['BBC', 'ENRON']
selected_dataset = col1.selectbox("Select dataset:", datasets)

if selected_dataset == 'BBC':
    df_text = load_data("bbc_text2.csv")
    df_responses = load_data("bbc_responses2.csv")
elif selected_dataset == 'ENRON':
    df_text = load_data("enron_text2.csv")
    df_responses = load_data("enron_responses2.csv")

file_names = df_text['file_name']
file_names = natsorted(file_names)
selected_file = col2.selectbox("Select file:", file_names)

anonymization_types = [
    "PII Deletion",
    "PII Labeling",
    "PII Replacement with Synthetic Data",
    "Diffractor (epsilon = 1)",
    "Diffractor (epsilon = 2)",
    "Diffractor (epsilon = 3)",
    "DP-Prompt (epsilon = 150)",
    "DP-Prompt (epsilon = 200)",
    "DP-Prompt (epsilon = 250)",
    "DP-MLM (epsilon = 50)",
    "DP-MLM (epsilon = 75)",
    "DP-MLM (epsilon = 100)",
]
anonymization_type_map = {
    "PII Deletion": "text_pii_deleted",
    "PII Labeling": "text_pii_labeled",
    "PII Replacement with Synthetic Data": "text_pii_synthetic",
    "Diffractor (epsilon = 1)": "text_pii_dp_diffractor1",
    "Diffractor (epsilon = 2)": "text_pii_dp_diffractor2",
    "Diffractor (epsilon = 3)": "text_pii_dp_diffractor3",
    "DP-Prompt (epsilon = 150)": "text_pii_dp_dp_prompt1",
    "DP-Prompt (epsilon = 200)": "text_pii_dp_dp_prompt2",
    "DP-Prompt (epsilon = 250)": "text_pii_dp_dp_prompt3",
    "DP-MLM (epsilon = 50)": "text_pii_dp_dpmlm1",
    "DP-MLM (epsilon = 75)": "text_pii_dp_dpmlm2",
    "DP-MLM (epsilon = 100)": "text_pii_dp_dpmlm3"
}

response_type_map = {
    key: val.replace("text_", "response_")
    for key, val in anonymization_type_map.items()
}

selected_anonymization = col3.selectbox("Select anonymization type:", anonymization_types)

### TEXT ANONYMISATION ###

col1, col2 = st.columns([1, 1])
text_row = df_text[df_text['file_name'] == selected_file].iloc[0]

col1.subheader(f"📄 Original text for `{selected_file}`")
col1.text_area(
    label="Original Text:",
    value=text_row['text_with_pii'],
    height=400,
    label_visibility="hidden" # visible, hidden, collapsed
)

col2.subheader(f"🔐 Text after {selected_anonymization} for `{selected_file}`")
col2.text_area(
    label="Anonymized Text:",
    value=text_row[anonymization_type_map[selected_anonymization]],
    height=400,
    label_visibility="hidden" # visible, hidden, collapsed
)

if selected_anonymization in ["PII Deletion", "PII Labeling", "PII Replacement with Synthetic Data"]:
    st_analyze_results = [RecognizerResult(**item) for item in json.loads(text_row['details'])]
    with st.expander("Personally Identifiable Information (PII) detected in the original text", expanded=False):
        annotated_tokens = annotate(text=text_row['text_with_pii'], analyze_results=st_analyze_results)
        annotated_text(*annotated_tokens)

    with st.expander("Detailed Findings", expanded=False):
        if text_row['details']:
            df_text = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
            df_text["Text"] = [text_row['text_with_pii'][res.start:res.end] for res in st_analyze_results]
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

### RAG ###

question_utility = """Please generate a concise and factual summary of the text. 
The summary should be structured into the following three key attributes:
[Attribute 1: TOPIC/CENTRAL IDEA]: main topic or central idea of the provided text
[Attribute 2: MAIN DETAILS ABOUT TOPIC/CENTRAL IDEA]: key events, discussion points, or details that support the central idea. For example, in a sports report, include scores and key plays; in a conversational email, note discussion topics and any actions mentioned.
[Attribute 3: IMPORTANT FACTS/EVENTS]: critical facts, evnets, data, or viewpoints that are essential to understanding the text

Please format your response as follows:
[Attribute 1: TOPIC/CENTRAL IDEA]:
[Attribute 2: MAIN DETAILS ABOUT TOPIC/CENTRAL IDEA]:
[Attribute 3: IMPORTANT FACTS/EVENTS]:
                    
Ensure that your summary:
- Is concise and uses clear, simple language.
- Remains factual and unbiased, without including personal opinions.
- Maintains a logical order and comprehensively covers the information provided in the text.
"""
question_privacy = "Please analyze the following text for any private or sensitive information, like personally identifiable information (PII). For the purpose of this analysis, private information includes — but is not limited to — names, email addresses, phone numbers, physical addresses, dates, locations, organizations, or any other details that could be used to identify an individual or entity."

response_row_utility = df_responses[(df_responses['file_name'] == selected_file) & (df_responses['question'] == question_utility)].iloc[0]
response_utility_with_pii = response_row_utility['response_with_pii']
response_utility_anonymized = response_row_utility[response_type_map[selected_anonymization]]

st.subheader(f"💬 Prompt for :green[utility] experiments")
st.text_area(
    label="Prompt for utility experiments", 
    value=question_utility, 
    height=200, 
    key="question_utility",
    label_visibility="hidden"
)

col1, col2 = st.columns([1, 1])

col1.subheader(f"📄 Response based on the original text")
col1.text_area(
    label="Response on the original text", 
    value=response_utility_with_pii, 
    height=200, 
    key="response_utility_with_pii",
    label_visibility="hidden" # visible, hidden, collapsed
)

col2.subheader(f"🔐 Response based on the text after {selected_anonymization}")
col2.text_area(
    label="Response on the anonymized text", 
    value=response_utility_anonymized, 
    height=200, 
    key="response_utility_anonymized",
    label_visibility="hidden" # visible, hidden, collapsed
)

response_row_privacy = df_responses[(df_responses['file_name'] == selected_file) & (df_responses['question'] == question_privacy)].iloc[0]
response_privacy_with_pii = response_row_privacy['response_with_pii']
response_privacy_anonymized = response_row_privacy[response_type_map[selected_anonymization]]

st.subheader(f"💬 Prompt for :green[privacy] experiments")
st.text_area(
    label="Prompt for privacy experiments", 
    value=question_privacy, 
    height=100, 
    key="question_privacy",
    label_visibility="hidden"
)

col1, col2 = st.columns([1, 1])

col1.subheader(f"📄 Response based on the original text")
col1.text_area(
    label="Response on the original text", 
    value=response_privacy_with_pii, 
    height=200, 
    key="response_privacy_with_pii",
    label_visibility="hidden" # visible, hidden, collapsed
)

col2.subheader(f"🔐 Response based on the text after {selected_anonymization}")
col2.text_area(
    label="Response on the anonymized text", 
    value=response_privacy_anonymized, 
    height=200, 
    key="response_privacy_anonymized",
    label_visibility="hidden" # visible, hidden, collapsed
)

### Evaluation ###

utility_eval = json.loads(response_row_utility['evaluation'])
privacy_eval = json.loads(response_row_privacy['evaluation'])

rouge1_score = round(utility_eval[response_type_map[selected_anonymization]]['rouge_score1'], 2)
rougeL_score = round(utility_eval[response_type_map[selected_anonymization]]['rouge_scoreL'], 2)
bleu_score = round(utility_eval[response_type_map[selected_anonymization]]['bleu_score'], 2)
cosine_similarity_score = round(utility_eval[response_type_map[selected_anonymization]]['cosine_similarity'], 2)
perplexity_score = round(utility_eval[response_type_map[selected_anonymization]]['perplexity'], 2)
llm_score = extract_llm_score(privacy_eval)

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

# Display evaluation results in two columns
col1, col2 = st.columns([1, 1])

# Evaluation table
col1.subheader(f"📊 Evaluation of the response based on the text after {selected_anonymization}")
col1.dataframe(evaluation_df)

# Bar chart visualization
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

