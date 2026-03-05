"""
Generate RAG responses: for each (file, question), query Pinecone by text_type, get LLM answer, insert into responses table.

If you hit OpenAI 429 (Too Many Requests), add --delay 1.0 or higher to throttle.

Where to run:
- Server: good if DB is on server or you want long-running jobs (nohup/screen/tmux). No GPU needed.
- Laptop: fine if DB is reachable (e.g. cloud) and you keep the process running.

Before first run for TAB, create the responses table:
  python -c "import sys; sys.path.insert(0,'src'); from Data.Database_management import create_table_responses; create_table_responses('tab_responses')"

Example (TAB, 7 text types in Pinecone):
  PYTHONPATH=src python src/RAG/Response_generation.py --tab --last 10201
"""
import argparse
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAG.Pinecone_LlamaIndex import getResponse
from Data.Database_management import retrieve_record_by_name, insert_responses
from constants import ORDERED_TEXT_TYPES, TAB_TEXT_TYPES

st_logger = logging.getLogger("Response generation")
st_logger.setLevel(logging.INFO)


def get_all_questions():
    questions = []

    # UTILITY
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
    questions.append(question_utility)
    
    # PRIVACY - UNTARGETED ATTACK
    question_privacy = "Please analyze the following text for any private or sensitive information, like personally identifiable information (PII). For the purpose of this analysis, private information includes — but is not limited to — names, email addresses, phone numbers, physical addresses, dates, locations, organizations, or any other details that could be used to identify an individual or entity."
    questions.append(question_privacy)

    return questions

def get_all_responses_entry(
    table_responses_name,
    index_name,
    file_name,
    database_file,
    text_types_to_query=None,
    delay_seconds: float = 0,
):
    """Query Pinecone for each text_type, get LLM response, insert one row per question. text_types_to_query: if set, only these are queried (others get "")."""
    questions = get_all_questions()
    file_hash = database_file["file_hash"]
    query_types = set(text_types_to_query) if text_types_to_query else None

    for question in questions:
        st_logger.info("file_name: %s", file_name)

        response_strings = []
        for text_type in ORDERED_TEXT_TYPES:
            if query_types is not None and text_type not in query_types:
                response_strings.append("")
                continue
            try:
                response_obj, _nodes, _eval = getResponse(index_name, question, [file_hash, text_type])
                response_strings.append(str(response_obj))
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            except Exception as e:
                st_logger.warning("getResponse failed %s %s: %s", file_name, text_type, e)
                response_strings.append("")

        insert_responses(
            table_responses_name,
            file_name,
            question,
            *response_strings,
            evaluation=None,
        )
        st_logger.info("Responses inserted for file_name=%s", file_name)



def get_all_responses_database(
    table_name,
    table_responses_name,
    index_name,
    file_name_pattern,
    start,
    last,
    text_types_to_query=None,
    skip_indices=None,
    delay_seconds: float = 0,
):
    """Iterate over file indices [start, last], get RAG responses and insert. skip_indices: e.g. {61} for Enron."""
    skip_indices = skip_indices or set()
    for i in range(start, last + 1):
        if table_name.startswith("enron") and i in skip_indices:
            continue
        file_name = file_name_pattern.format(i)
        database_file = retrieve_record_by_name(table_name, file_name)
        if database_file is None:
            st_logger.warning("File not found: %s", file_name)
            continue
        get_all_responses_entry(
            table_responses_name=table_responses_name,
            index_name=index_name,
            file_name=file_name,
            database_file=database_file,
            text_types_to_query=text_types_to_query,
            delay_seconds=delay_seconds,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG responses and insert into responses table.")
    parser.add_argument("--table", default="tab_text2", help="Text table (default: tab_text2)")
    parser.add_argument("--responses-table", default="tab_responses", help="Responses table (default: tab_responses)")
    parser.add_argument("--index", default="tab", help="Pinecone index (default: tab)")
    parser.add_argument("--file-name-pattern", default="TAB_{}", help="File name pattern (default: TAB_{})")
    parser.add_argument("--start", type=int, default=1, help="First file index")
    parser.add_argument("--last", type=int, required=True, help="Last file index (inclusive)")
    parser.add_argument("--tab", action="store_true", help="Only 7 text types in Pinecone (no dp_prompt/dpmlm)")
    parser.add_argument("--delay", type=float, default=0, metavar="SECONDS", help="Sleep after each getResponse to avoid OpenAI 429 (e.g. 1.0)")
    args = parser.parse_args()
    text_types = TAB_TEXT_TYPES if args.tab else None
    get_all_responses_database(
        table_name=args.table,
        table_responses_name=args.responses_table,
        index_name=args.index,
        file_name_pattern=args.file_name_pattern,
        start=args.start,
        last=args.last,
        text_types_to_query=text_types,
        skip_indices={61} if args.table.startswith("enron") else None,
        delay_seconds=args.delay,
    )