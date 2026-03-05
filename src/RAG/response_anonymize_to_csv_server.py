"""
Server-side script to run anonymization on RAG responses and save results to CSV.
Supports Presidio and Diffractor methods.

Input CSV: file_name, question, response_with_pii
Output CSV: file_name, question, response_with_pii, + 3 anonymized response columns

Usage:
  python response_anonymize_to_csv_server.py --method presidio --input ./responses_export.csv --output ./presidio_response_results.csv
  python response_anonymize_to_csv_server.py --method diffractor --input ./responses_export.csv --output ./diffractor_response_results.csv
  python response_anonymize_to_csv_server.py --method diffractor --start 1 --end 100
"""
import argparse
import csv
import gc
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st_logger = logging.getLogger("Response_Anonymize_CSV_Server")
st_logger.setLevel(logging.INFO)
if not st_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    st_logger.addHandler(handler)

KEY_COLUMNS = ["file_name", "question"]


def _anonymize_presidio(response_text):
    """Run Presidio anonymization (delete, label, synthetic) on a response."""
    from Presidio.Presidio import delete_pii, label_pii, replace_pii
    return {
        "response_postprocessed_pii_deleted": delete_pii(response_text),
        "response_postprocessed_pii_labeled": label_pii(response_text),
        "response_postprocessed_pii_synthetic": replace_pii(response_text),
    }


def _anonymize_diffractor(response_text):
    """Run Diffractor DP on a response with epsilon 1, 2, 3."""
    from Differential_privacy.DP import diff_privacy_diffractor
    return {
        "response_postprocessed_pii_diffractor1": diff_privacy_diffractor(response_text, epsilon=1),
        "response_postprocessed_pii_diffractor2": diff_privacy_diffractor(response_text, epsilon=2),
        "response_postprocessed_pii_diffractor3": diff_privacy_diffractor(response_text, epsilon=3),
    }


METHOD_REGISTRY = {
    "presidio": {
        "func": _anonymize_presidio,
        "columns": [
            "response_postprocessed_pii_deleted",
            "response_postprocessed_pii_labeled",
            "response_postprocessed_pii_synthetic",
        ],
        "default_output": "./presidio_response_results.csv",
    },
    "diffractor": {
        "func": _anonymize_diffractor,
        "columns": [
            "response_postprocessed_pii_diffractor1",
            "response_postprocessed_pii_diffractor2",
            "response_postprocessed_pii_diffractor3",
        ],
        "default_output": "./diffractor_response_results.csv",
    },
}


def load_already_processed_keys(output_csv_path):
    """Load set of (file_name, question) already present in output CSV."""
    processed = set()
    if not os.path.exists(output_csv_path):
        return processed
    try:
        with open(output_csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "file_name" not in reader.fieldnames:
                return processed
            for row in reader:
                key = tuple((row.get(c) or "").strip() for c in KEY_COLUMNS)
                processed.add(key)
        st_logger.info(f"Loaded {len(processed)} already-processed (file_name, question) from {output_csv_path}")
    except Exception as e:
        st_logger.warning(f"Could not read existing output for skip-duplicates: {e}")
    return processed


def process_and_save_to_csv(method, input_csv_path, output_csv_path, start_row=1, end_row=None):
    """
    Process responses from input CSV with the specified anonymization method and save to output CSV.

    Parameters:
    - method: 'presidio' or 'diffractor'
    - input_csv_path: Path to input CSV with columns file_name, question, response_with_pii
    - output_csv_path: Path to output CSV file
    - start_row: Row number to start processing from (1-indexed; 1 = first data row after header)
    - end_row: Row number to stop processing at (None = process all)
    """
    config = METHOD_REGISTRY[method]
    anonymize_func = config["func"]
    result_columns = config["columns"]
    output_columns = ["file_name", "question", "response_with_pii"] + result_columns

    file_exists = os.path.exists(output_csv_path)
    already_processed = load_already_processed_keys(output_csv_path)

    with open(
        input_csv_path, mode="r", newline="", encoding="utf-8"
    ) as input_file, open(
        output_csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8",
    ) as output_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or "response_with_pii" not in (reader.fieldnames or []):
            st_logger.error("Input CSV must have columns: file_name, question, response_with_pii")
            return

        writer = csv.DictWriter(output_file, fieldnames=output_columns)
        if not file_exists:
            writer.writeheader()
            st_logger.info(f"Created new CSV: {output_csv_path}")
        else:
            st_logger.info(f"Appending to existing CSV: {output_csv_path}")

        for row_number, row in enumerate(reader, start=2):
            if row_number < start_row:
                continue
            if end_row is not None and row_number > end_row:
                break

            file_name = (row.get("file_name") or "").strip()
            question = (row.get("question") or "").strip()
            response_with_pii = (row.get("response_with_pii") or "").strip()

            key = (file_name, question)
            if key in already_processed:
                st_logger.info(f"Row {row_number}: already in output, skipping (file_name={file_name!r})")
                continue

            if not response_with_pii:
                st_logger.warning(f"Row {row_number}: empty response_with_pii, skipping")
                continue

            st_logger.info(f"Processing row {row_number} file_name={file_name!r}")

            try:
                results = anonymize_func(response_with_pii)
                out_row = {"file_name": file_name, "question": question, "response_with_pii": response_with_pii}
                out_row.update(results)
                writer.writerow(out_row)
                output_file.flush()
                st_logger.info(f"Row {row_number} processed and saved")
            except Exception as e:
                st_logger.error(f"Error at row {row_number}: {e}")
                out_row = {"file_name": file_name, "question": question, "response_with_pii": response_with_pii}
                out_row[result_columns[0]] = f"ERROR: {e}"
                for col in result_columns[1:]:
                    out_row[col] = ""
                writer.writerow(out_row)
                output_file.flush()

            already_processed.add(key)
            gc.collect()

    st_logger.info(f"{method} response processing complete. Output: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run anonymization on RAG responses and save to CSV."
    )
    parser.add_argument("--method", required=True, choices=list(METHOD_REGISTRY.keys()),
                        help="Anonymization method")
    parser.add_argument("--input", default="./responses_export.csv",
                        help="Input CSV (file_name, question, response_with_pii)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: method-specific)")
    parser.add_argument("--start", type=int, default=1, help="First row to process (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="Last row to process (inclusive)")
    args = parser.parse_args()

    output_path = args.output or METHOD_REGISTRY[args.method]["default_output"]
    process_and_save_to_csv(args.method, args.input, output_path, start_row=args.start, end_row=args.end)
