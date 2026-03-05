from dotenv import load_dotenv
import os
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") # e.g., "postgresql://user:password@localhost/dbname"

def create_table_text(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    pdf_bytes BYTEA,
                    text_with_pii TEXT,
                    text_pii_deleted TEXT,
                    text_pii_labeled TEXT, 
                    text_pii_synthetic TEXT,
                    text_pii_dp_diffractor1 TEXT,
                    text_pii_dp_diffractor2 TEXT,
                    text_pii_dp_diffractor3 TEXT,
                    text_pii_dp_dp_prompt1 TEXT,
                    text_pii_dp_dp_prompt2 TEXT,
                    text_pii_dp_dp_prompt3 TEXT,
                    text_pii_dp_dpmlm1 TEXT,
                    text_pii_dp_dpmlm2 TEXT,
                    text_pii_dp_dpmlm3 TEXT,
                    details TEXT
                );
            """)
            conn.commit()
            print(f"Table '{table_name}' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def create_table_responses(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    question TEXT,
                    response_with_pii TEXT,
                    response_pii_deleted TEXT,
                    response_pii_labeled TEXT, 
                    response_pii_synthetic TEXT,
                    response_pii_dp_diffractor1 TEXT,
                    response_pii_dp_diffractor2 TEXT,
                    response_pii_dp_diffractor3 TEXT,
                    response_pii_dp_dp_prompt1 TEXT,
                    response_pii_dp_dp_prompt2 TEXT,
                    response_pii_dp_dp_prompt3 TEXT,
                    response_pii_dp_dpmlm1 TEXT,
                    response_pii_dp_dpmlm2 TEXT,
                    response_pii_dp_dpmlm3 TEXT,
                    evaluation JSONB
                );
            """)
            conn.commit()
            print(f"Table '{table_name}' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def create_table_responses_postprocessed(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    question TEXT,
                    response_with_pii TEXT,
                    response_postprocessed_pii_deleted TEXT,
                    response_postprocessed_pii_labeled TEXT, 
                    response_postprocessed_pii_synthetic TEXT,
                    response_postprocessed_pii_diffractor1 TEXT,
                    response_postprocessed_pii_diffractor2 TEXT,
                    response_postprocessed_pii_diffractor3 TEXT,
                    response_postprocessed_pii_dp_prompt1 TEXT,
                    response_postprocessed_pii_dp_prompt2 TEXT,
                    response_postprocessed_pii_dp_prompt3 TEXT,
                    response_postprocessed_pii_dpmlm1 TEXT,
                    response_postprocessed_pii_dpmlm2 TEXT,
                    response_postprocessed_pii_dpmlm3 TEXT,
                    evaluation JSONB
                );
            """)
            conn.commit()
            print(f"Table '{table_name}' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def delete_table(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            conn.commit()
            print(f"Table '{table_name}' deleted successfully")
    except Exception as e:
        print(f"Error deleting table: {e}")
    finally:
        conn.close()

def export_table_to_csv(table_name, csv_file_path):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            with open(csv_file_path, 'w', encoding='utf-8') as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH CSV HEADER", f)
        print(f"Table '{table_name}' exported to '{csv_file_path}' successfully.")
    except Exception as e:
        print(f"Error exporting table '{table_name}': {e}")
    finally:
        conn.close()
    
def list_records(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT id, file_name, file_hash, uploaded_at FROM {table_name} ORDER BY uploaded_at DESC")
            return cur.fetchall()
    except Exception as e:
        print(f"Error listing the records in table: {e}")
    finally:
        conn.close()


def list_file_names_ordered_by_number(table_name):
    """
    Return list of file_name values ordered by the numeric suffix (e.g. TAB_1, TAB_2, ...).
    Useful for tab_text2 and other tables where file_name is like PREFIX_N.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT file_name FROM {table_name}
                ORDER BY COALESCE((regexp_match(file_name, '[0-9]+$'))[1]::integer, 0) ASC
            """)
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"Error listing file names: {e}")
        return []
    finally:
        conn.close()


def retrieve_record_by_name(table_name, file_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_name = %s", (file_name,))
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving record by name: {e}")
    finally:
        conn.close()
    
def retrieve_record_by_hash(table_name, file_hash):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_hash = %s", (file_hash,))
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving record by hash: {e}")
    finally:
        conn.close()

def insert_record(table_name, file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, text_pii_dp_dp_prompt1, text_pii_dp_dp_prompt2, text_pii_dp_dp_prompt3, text_pii_dp_dpmlm1, text_pii_dp_dpmlm2, text_pii_dp_dpmlm3, details):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, text_pii_dp_dp_prompt1, text_pii_dp_dp_prompt2, text_pii_dp_dp_prompt3, text_pii_dp_dpmlm1, text_pii_dp_dpmlm2, text_pii_dp_dpmlm3, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, file_hash, psycopg2.Binary(pdf_bytes), text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, text_pii_dp_dp_prompt1, text_pii_dp_dp_prompt2, text_pii_dp_dp_prompt3, text_pii_dp_dpmlm1, text_pii_dp_dpmlm2, text_pii_dp_dpmlm3, details)
            )
            conn.commit()
            print(f"Record inserted into '{table_name}' successfully")
    except Exception as e:
        print(f"Error inserting record: {e}")
    finally:
        conn.close()

def insert_partial_record(table_name, file_name, file_hash, pdf_bytes, text_with_pii=None, text_pii_deleted=None, text_pii_labeled=None, text_pii_synthetic=None, text_pii_dp_diffractor1=None, text_pii_dp_diffractor2=None, text_pii_dp_diffractor3=None, text_pii_dp_dp_prompt1=None, text_pii_dp_dp_prompt2=None, text_pii_dp_dp_prompt3=None, text_pii_dp_dpmlm1=None, text_pii_dp_dpmlm2=None, text_pii_dp_dpmlm3=None, details=None):
    """
    Insert a record into the specified table with only the provided data.
    
    Parameters:
    - table_name: The name of the table to insert data into
    - Other parameters: Optional data to insert into the table
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            columns = ["file_name", "file_hash", "pdf_bytes"]
            values = [file_name, file_hash, pdf_bytes]
            placeholders = ["%s", "%s", "%s"]

            # Collect only the provided columns and values
            if text_with_pii is not None:
                columns.append("text_with_pii")
                values.append(text_with_pii)
                placeholders.append("%s")
            if text_pii_deleted is not None:
                columns.append("text_pii_deleted")
                values.append(text_pii_deleted)
                placeholders.append("%s")
            if text_pii_labeled is not None:
                columns.append("text_pii_labeled")
                values.append(text_pii_labeled)
                placeholders.append("%s")
            if text_pii_synthetic is not None:
                columns.append("text_pii_synthetic")
                values.append(text_pii_synthetic)
                placeholders.append("%s")
            if text_pii_dp_diffractor1 is not None:
                columns.append("text_pii_dp_diffractor1")
                values.append(text_pii_dp_diffractor1)
                placeholders.append("%s")
            if text_pii_dp_diffractor2 is not None:
                columns.append("text_pii_dp_diffractor2")
                values.append(text_pii_dp_diffractor2)
                placeholders.append("%s")
            if text_pii_dp_diffractor3 is not None:
                columns.append("text_pii_dp_diffractor3")
                values.append(text_pii_dp_diffractor3)
                placeholders.append("%s")
            if text_pii_dp_dp_prompt1 is not None:
                columns.append("text_pii_dp_dp_prompt1")
                values.append(text_pii_dp_dp_prompt1)
                placeholders.append("%s")
            if text_pii_dp_dp_prompt2 is not None:
                columns.append("text_pii_dp_dp_prompt2")
                values.append(text_pii_dp_dp_prompt2)
                placeholders.append("%s")
            if text_pii_dp_dp_prompt3 is not None:
                columns.append("text_pii_dp_dp_prompt3")
                values.append(text_pii_dp_dp_prompt3)
                placeholders.append("%s")
            if text_pii_dp_dpmlm1 is not None:
                columns.append("text_pii_dp_dpmlm1")
                values.append(text_pii_dp_dpmlm1)
                placeholders.append("%s")
            if text_pii_dp_dpmlm2 is not None:
                columns.append("text_pii_dp_dpmlm2")
                values.append(text_pii_dp_dpmlm2)
                placeholders.append("%s")
            if text_pii_dp_dpmlm3 is not None:
                columns.append("text_pii_dp_dpmlm3")
                values.append(text_pii_dp_dpmlm3)
                placeholders.append("%s")
            if details is not None:
                columns.append("details")
                values.append(details)
                placeholders.append("%s")

            # Construct the SQL query dynamically
            cur.execute(
                f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})",
                values
            )
            conn.commit()
            print(f"Partial record inserted into '{table_name}' successfully")
    except Exception as e:
        print(f"Error inserting partial record: {e}")
    finally:
        conn.close()

def insert_responses(table_name, file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, evaluation):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, evaluation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, evaluation)
            )
            conn.commit()
            print(f"Response inserted into '{table_name}' successfully")
    except Exception as e:
        print(f"Error inserting responses: {e}")
    finally:
        conn.close()

def retrieve_responses_by_name(table_name, file_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_name = %s", (file_name,))
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving record by name: {e}")
    finally:
        conn.close()

def retrieve_responses_by_name_and_question(table_name, file_name, question):
    """
    Return the row for (file_name, question). Question is matched after trimming both sides,
    so minor whitespace/newline differences between DB and code do not cause "No data found".
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            q = question.strip() if question else ""
            cur.execute(
                f"SELECT * FROM {table_name} WHERE file_name = %s AND TRIM(question) = %s",
                (file_name, q),
            )
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving responses by name and question: {e}")
    finally:
        conn.close()


def count_response_rows(table_name, file_name_like=None):
    """
    Return total row count for a responses table. If file_name_like is set (e.g. 'TAB_%'),
    count only rows matching that pattern. Useful to check which table has data before evaluation.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            if file_name_like:
                cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE file_name LIKE %s", (file_name_like,))
            else:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cur.fetchone()[0]
    except Exception as e:
        print(f"Error counting rows in {table_name}: {e}")
        return None
    finally:
        conn.close()


def list_file_names_for_question(table_name, question, file_name_like=None):
    """
    Return list of file_name values that have a row with the given question (matched after TRIM).
    Optionally filter by file_name pattern (e.g. 'TAB_%'). Ordered by numeric suffix if pattern like PREFIX_N.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            q = question.strip() if question else ""
            if file_name_like:
                cur.execute(
                    f"""SELECT file_name FROM {table_name}
                       WHERE TRIM(question) = %s AND file_name LIKE %s
                       ORDER BY COALESCE((regexp_match(file_name, '[0-9]+$'))[1]::integer, 0) ASC""",
                    (q, file_name_like),
                )
            else:
                cur.execute(
                    f"""SELECT file_name FROM {table_name} WHERE TRIM(question) = %s
                       ORDER BY file_name""",
                    (q,),
                )
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"Error listing file_name for question in {table_name}: {e}")
        return []
    finally:
        conn.close()


def retrieve_all_responses(table_name):
    """
    Retrieve all rows from a responses table with columns file_name, question, response_with_pii.
    Used for exporting responses to CSV before Presidio/Diffractor processing on the server.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT file_name, question, response_with_pii FROM {table_name} ORDER BY id"
            )
            return cur.fetchall()
    except Exception as e:
        print(f"Error retrieving all responses from {table_name}: {e}")
        return []
    finally:
        conn.close()


def retrieve_response_rows_by_file_name(table_name, file_name):
    """
    Return all response rows for a given file_name (one row per question).
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_name = %s ORDER BY id", (file_name,))
            return cur.fetchall()
    except Exception as e:
        print(f"Error retrieving response rows for {file_name}: {e}")
        return []
    finally:
        conn.close()


def update_response_columns(table_name, file_name, question, **kwargs):
    """
    Update only the specified columns for a row in a responses table.
    Row is identified by (file_name, question). Use for filling dp_prompt / dpmlm response columns.

    Parameters:
    - table_name: The name of the responses table (e.g. tab_responses)
    - file_name: The file name to identify the row
    - question: The question to identify the row
    - **kwargs: Column names and values to update (e.g. response_pii_dp_dp_prompt1=..., response_pii_dp_dpmlm1=...)
    """
    if not kwargs:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            set_clause = ", ".join([f"{key} = %s" for key in kwargs.keys()])
            values = list(kwargs.values()) + [file_name, question]
            cur.execute(
                f"UPDATE {table_name} SET {set_clause} WHERE file_name = %s AND question = %s",
                values
            )
            conn.commit()
            if cur.rowcount > 0:
                columns_str = ", ".join(sorted(kwargs.keys()))
                print(f"Updated response columns for file_name={file_name!r}: {columns_str}")
    except Exception as e:
        print(f"Error updating response columns: {e}")
    finally:
        conn.close()


def add_data(table_name, file_hash, **kwargs):
    """
    Update specific columns in an existing record identified by file_hash.
    
    Parameters:
    - table_name: The name of the table
    - file_hash: The hash used to identify the record
    - **kwargs: Column names and values to update
    """
    try:
        if not kwargs:
            print("No columns specified for update")
            return
            
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # Construct SET clause for the UPDATE statement
            set_clause = ", ".join([f"{key} = %s" for key in kwargs.keys()])
            values = list(kwargs.values())
            values.append(file_hash)  # Add file_hash for the WHERE clause
            
            # Execute the UPDATE statement
            cur.execute(
                f"UPDATE {table_name} SET {set_clause} WHERE file_hash = %s",
                values
            )
            rows_updated = cur.rowcount
            conn.commit()
            
            if rows_updated > 0:
                print(f"Record with file_hash '{file_hash}' updated successfully")
            else:
                print(f"No record found with file_hash '{file_hash}'")
                
    except Exception as e:
        print(f"Error updating record: {e}")
    finally:
        conn.close()

def update_response_evaluation(table_name, file_name, question, evaluation):
    """
    Update the evaluation column for the row identified by file_name and question.
    Question is matched with TRIM so whitespace differences between DB and code do not prevent updates.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            q = question.strip() if question else ""
            cur.execute(f"""
                UPDATE {table_name}
                SET evaluation = %s
                WHERE file_name = %s AND TRIM(question) = %s;
            """, (evaluation, file_name, q))
            conn.commit()
            if cur.rowcount == 0:
                print(f"Warning: no row updated for file_name={file_name!r} and question (trimmed); evaluation not saved.")
            else:
                print(f"Evaluation updated successfully for file: {file_name} (1 row)")
    except Exception as e:
        print(f"Error updating evaluation: {e}")
    finally:
        conn.close()


def update_responses_preprocessed_columns(table_name, file_name, question, **kwargs):
    """
    Update only the specified columns for a row in a responses_postprocessed table.
    Row is identified by (file_name, question). Use for loading Presidio-only or Diffractor-only CSVs.

    Parameters:
    - table_name: The name of the postprocessed table
    - file_name: The file name to identify the row
    - question: The question to identify the row
    - **kwargs: Column names and values to update (e.g. response_postprocessed_pii_deleted=..., response_postprocessed_pii_labeled=..., response_postprocessed_pii_synthetic=...)
    """
    if not kwargs:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            set_clause = ", ".join([f"{key} = %s" for key in kwargs.keys()])
            values = list(kwargs.values()) + [file_name, question]
            cur.execute(
                f"UPDATE {table_name} SET {set_clause} WHERE file_name = %s AND question = %s",
                values
            )
            conn.commit()
            if cur.rowcount > 0:
                print(f"Updated response postprocessed columns for file_name={file_name!r}")
    except Exception as e:
        print(f"Error updating responses preprocessed columns: {e}")
    finally:
        conn.close()


def insert_responses_preprocessed(table_name, file_name, question, response_with_pii, response_postprocessed_pii_deleted, response_postprocessed_pii_labeled, response_postprocessed_pii_synthetic, response_postprocessed_pii_diffractor1, response_postprocessed_pii_diffractor2, response_postprocessed_pii_diffractor3, response_postprocessed_pii_dp_prompt1, response_postprocessed_pii_dp_prompt2, response_postprocessed_pii_dp_prompt3, response_postprocessed_pii_dpmlm1, response_postprocessed_pii_dpmlm2, response_postprocessed_pii_dpmlm3, evaluation):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, question, response_with_pii, response_postprocessed_pii_deleted, response_postprocessed_pii_labeled, response_postprocessed_pii_synthetic, response_postprocessed_pii_diffractor1, response_postprocessed_pii_diffractor2, response_postprocessed_pii_diffractor3, response_postprocessed_pii_dp_prompt1, response_postprocessed_pii_dp_prompt2, response_postprocessed_pii_dp_prompt3, response_postprocessed_pii_dpmlm1, response_postprocessed_pii_dpmlm2, response_postprocessed_pii_dpmlm3, evaluation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, question, response_with_pii, response_postprocessed_pii_deleted, response_postprocessed_pii_labeled, response_postprocessed_pii_synthetic, response_postprocessed_pii_diffractor1, response_postprocessed_pii_diffractor2, response_postprocessed_pii_diffractor3, response_postprocessed_pii_dp_prompt1, response_postprocessed_pii_dp_prompt2, response_postprocessed_pii_dp_prompt3, response_postprocessed_pii_dpmlm1, response_postprocessed_pii_dpmlm2, response_postprocessed_pii_dpmlm3, evaluation)
            )
            conn.commit()
            print(f"Preprocessed response inserted into '{table_name}' successfully")
    except Exception as e:
        print(f"Error inserting preprocessed responses: {e}")
    finally:
        conn.close()

def merge_tables_into_new(source_table1, source_table2, new_table_name):
    """
    Merge two tables with the same structure into a new table.
    First inserts all records from source_table1, then inserts Enron_100 from source_table2,
    followed by all remaining records from source_table2.
    
    Parameters:
    - source_table1: The name of the first source table
    - source_table2: The name of the second source table
    - new_table_name: The name of the new table to create and merge into
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # First create the new table with the same structure
            cur.execute(f"""
                CREATE TABLE {new_table_name} AS 
                SELECT * FROM {source_table1} 
                WHERE 1=0;
            """)
            
            # Add primary key constraint on id
            cur.execute(f"""
                ALTER TABLE {new_table_name}
                ADD PRIMARY KEY (id);
            """)
            
            # Insert records from first source table
            cur.execute(f"""
                INSERT INTO {new_table_name}
                SELECT * FROM {source_table1};
            """)
            
            # Get the max id from the first table
            cur.execute(f"SELECT MAX(id) FROM {source_table1}")
            max_id = cur.fetchone()[0]
            
            # First insert Enron_100 from second table
            cur.execute(f"""
                INSERT INTO {new_table_name}
                SELECT 
                    99,
                    file_name,
                    file_hash,
                    pdf_bytes,
                    text_with_pii,
                    text_pii_deleted,
                    text_pii_labeled,
                    text_pii_synthetic,
                    text_pii_dp_diffractor1,
                    text_pii_dp_diffractor2,
                    text_pii_dp_diffractor3,
                    text_pii_dp_dp_prompt1,
                    text_pii_dp_dp_prompt2,
                    text_pii_dp_dp_prompt3,
                    text_pii_dp_dpmlm1,
                    text_pii_dp_dpmlm2,
                    text_pii_dp_dpmlm3,
                    details
                FROM {source_table2}
                WHERE file_name = 'Enron_100';
            """)
            
            # Update max_id after inserting Enron_100
            cur.execute(f"SELECT MAX(id) FROM {new_table_name}")
            max_id = cur.fetchone()[0]
            
            # Insert remaining records from second table
            cur.execute(f"""
                INSERT INTO {new_table_name}
                SELECT 
                    id + {max_id},
                    file_name,
                    file_hash,
                    pdf_bytes,
                    text_with_pii,
                    text_pii_deleted,
                    text_pii_labeled,
                    text_pii_synthetic,
                    text_pii_dp_diffractor1,
                    text_pii_dp_diffractor2,
                    text_pii_dp_diffractor3,
                    text_pii_dp_dp_prompt1,
                    text_pii_dp_dp_prompt2,
                    text_pii_dp_dp_prompt3,
                    text_pii_dp_dpmlm1,
                    text_pii_dp_dpmlm2,
                    text_pii_dp_dpmlm3,
                    details
                FROM {source_table2}
                WHERE file_name != 'Enron_100';
            """)
            
            conn.commit()
            print(f"Successfully created new table '{new_table_name}' with merged data from '{source_table1}' and '{source_table2}'")
    except Exception as e:
        print(f"Error merging tables into new table: {e}")
    finally:
        conn.close()

def update_summary_question(table_name):
    """
    Updates the question in the response table from the old version to the new version,
    fixing the typo in 'evnets' to 'events'.
    
    Parameters:
    - table_name: The name of the response table to update
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            old_question = """Please generate a concise and factual summary of the text. 
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

            new_question = """Please generate a concise and factual summary of the text. 
The summary should be structured into the following three key attributes:
[Attribute 1: TOPIC/CENTRAL IDEA]: main topic or central idea of the provided text
[Attribute 2: MAIN DETAILS ABOUT TOPIC/CENTRAL IDEA]: key events, discussion points, or details that support the central idea. For example, in a sports report, include scores and key plays; in a conversational email, note discussion topics and any actions mentioned.
[Attribute 3: IMPORTANT FACTS/EVENTS]: critical facts, events, data, or viewpoints that are essential to understanding the text

Please format your response as follows:
[Attribute 1: TOPIC/CENTRAL IDEA]:
[Attribute 2: MAIN DETAILS ABOUT TOPIC/CENTRAL IDEA]:
[Attribute 3: IMPORTANT FACTS/EVENTS]:
                    
Ensure that your summary:
- Is concise and uses clear, simple language.
- Remains factual and unbiased, without including personal opinions.
- Maintains a logical order and comprehensively covers the information provided in the text.
"""

            # Update all rows where the question matches the old version
            cur.execute(f"""
                UPDATE {table_name}
                SET question = %s
                WHERE question = %s;
            """, (new_question, old_question))
            
            rows_updated = cur.rowcount
            conn.commit()
            print(f"Updated {rows_updated} rows in table '{table_name}'")
            
    except Exception as e:
        print(f"Error updating summary question: {e}")
    finally:
        conn.close()

def merge_response_tables_into_new(source_table1, source_table2, new_table_name):
    """
    Merge two response tables with the same structure into a new table.
    First inserts all records from source_table1, then inserts all Enron_100 records from source_table2,
    followed by all remaining records from source_table2.
    Finally renumbers all rows with sequential IDs.
    
    Parameters:
    - source_table1: The name of the first source response table
    - source_table2: The name of the second source response table
    - new_table_name: The name of the new table to create and merge into
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # First create the new table with the same structure
            cur.execute(f"""
                CREATE TABLE {new_table_name} AS 
                SELECT * FROM {source_table1} 
                WHERE 1=0;
            """)
            
            # Add primary key constraint on id
            cur.execute(f"""
                ALTER TABLE {new_table_name}
                ADD PRIMARY KEY (id);
            """)
            
            # Insert records from first source table
            cur.execute(f"""
                INSERT INTO {new_table_name}
                SELECT * FROM {source_table1};
            """)
            
            # Get the max id from the first table
            cur.execute(f"SELECT MAX(id) FROM {source_table1}")
            max_id = cur.fetchone()[0]
            
            # Insert all Enron_100 records from second table
            cur.execute(f"""
                INSERT INTO {new_table_name}
                SELECT 
                    id - {11},
                    file_name,
                    question,
                    response_with_pii,
                    response_pii_deleted,
                    response_pii_labeled,
                    response_pii_synthetic,
                    response_pii_dp_diffractor1,
                    response_pii_dp_diffractor2,
                    response_pii_dp_diffractor3,
                    response_pii_dp_dp_prompt1,
                    response_pii_dp_dp_prompt2,
                    response_pii_dp_dp_prompt3,
                    response_pii_dp_dpmlm1,
                    response_pii_dp_dpmlm2,
                    response_pii_dp_dpmlm3,
                    evaluation
                FROM {source_table2}
                WHERE file_name = 'Enron_100';
            """)
            
            # Update max_id after inserting Enron_100
            cur.execute(f"SELECT MAX(id) FROM {new_table_name}")
            max_id = cur.fetchone()[0]
            
            # Insert remaining records from second table
            cur.execute(f"""
                INSERT INTO {new_table_name}
                SELECT 
                    id + {max_id-1},
                    file_name,
                    question,
                    response_with_pii,
                    response_pii_deleted,
                    response_pii_labeled,
                    response_pii_synthetic,
                    response_pii_dp_diffractor1,
                    response_pii_dp_diffractor2,
                    response_pii_dp_diffractor3,
                    response_pii_dp_dp_prompt1,
                    response_pii_dp_dp_prompt2,
                    response_pii_dp_dp_prompt3,
                    response_pii_dp_dpmlm1,
                    response_pii_dp_dpmlm2,
                    response_pii_dp_dpmlm3,
                    evaluation
                FROM {source_table2}
                WHERE file_name != 'Enron_100';
            """)
            
            # Create a temporary table with the same structure
            cur.execute(f"""
                CREATE TABLE temp_{new_table_name} AS 
                SELECT * FROM {new_table_name} 
                WHERE 1=0;
            """)
            
            # Add primary key constraint on id
            cur.execute(f"""
                ALTER TABLE temp_{new_table_name}
                ADD PRIMARY KEY (id);
            """)
            
            # Insert all records with new sequential IDs
            cur.execute(f"""
                INSERT INTO temp_{new_table_name}
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY id),
                    file_name,
                    question,
                    response_with_pii,
                    response_pii_deleted,
                    response_pii_labeled,
                    response_pii_synthetic,
                    response_pii_dp_diffractor1,
                    response_pii_dp_diffractor2,
                    response_pii_dp_diffractor3,
                    response_pii_dp_dp_prompt1,
                    response_pii_dp_dp_prompt2,
                    response_pii_dp_dp_prompt3,
                    response_pii_dp_dpmlm1,
                    response_pii_dp_dpmlm2,
                    response_pii_dp_dpmlm3,
                    evaluation
                FROM {new_table_name};
            """)
            
            # Drop the original table and rename the temporary table
            cur.execute(f"DROP TABLE {new_table_name};")
            cur.execute(f"ALTER TABLE temp_{new_table_name} RENAME TO {new_table_name};")
            
            conn.commit()
            print(f"Successfully created new table '{new_table_name}' with merged data from '{source_table1}' and '{source_table2}' and renumbered IDs")
    except Exception as e:
        print(f"Error merging response tables into new table: {e}")
    finally:
        conn.close()
      
def merge_postprocessed_tables_into_new(source_table1, source_table2, new_table_name):
    """
    Merge two postprocessed response tables into a new table, ordered by the numerical part of file_name (ascending)
    and question (descending), with sequential IDs.
    
    Parameters:
    - source_table1: The name of the first source postprocessed table
    - source_table2: The name of the second source postprocessed table
    - new_table_name: The name of the new table to create and merge into
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # Create the new table with the same structure
            cur.execute(f"""
                CREATE TABLE {new_table_name} AS 
                SELECT * FROM {source_table1} 
                WHERE 1=0;
            """)
            
            # Add primary key constraint on id
            cur.execute(f"""
                ALTER TABLE {new_table_name}
                ADD PRIMARY KEY (id);
            """)
            
            # Insert all records from both tables with new sequential IDs
            cur.execute(f"""
                INSERT INTO {new_table_name}
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY substring(file_name from '_(\\d+)') ::integer ASC, question DESC),
                    file_name,
                    question,
                    response_with_pii,
                    response_postprocessed_pii_deleted,
                    response_postprocessed_pii_labeled,
                    response_postprocessed_pii_synthetic,
                    response_postprocessed_pii_diffractor1,
                    response_postprocessed_pii_diffractor2,
                    response_postprocessed_pii_diffractor3,
                    response_postprocessed_pii_dp_prompt1,
                    response_postprocessed_pii_dp_prompt2,
                    response_postprocessed_pii_dp_prompt3,
                    response_postprocessed_pii_dpmlm1,
                    response_postprocessed_pii_dpmlm2,
                    response_postprocessed_pii_dpmlm3,
                    evaluation
                FROM (
                    SELECT * FROM {source_table1}
                    UNION ALL
                    SELECT * FROM {source_table2}
                ) combined_tables
                ORDER BY substring(file_name from '_(\\d+)') ::integer ASC, question DESC;
            """)
            
            conn.commit()
            print(f"Successfully created new table '{new_table_name}' with merged data from '{source_table1}' and '{source_table2}' ordered by the numerical part of file_name (ascending) and question (descending)")
    except Exception as e:
        print(f"Error merging postprocessed tables: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_table_text("tab_text2")
    """
    create_table_text("enron_text2")
    create_table_text("bbc_text2")
    create_table_responses("enron_responses2")
    create_table_responses("bbc_responses2")
    create_table_text("enron_text3")
    create_table_responses("enron_responses3")

    export_table_to_csv("enron_text2", "./enron_text2.csv")
    export_table_to_csv("bbc_text2", "/./bbc_text2.csv")
    export_table_to_csv("enron_responses2", "/./enron_responses2.csv")
    export_table_to_csv("bbc_responses2", "/./bbc_responses2.csv")

    create_table_responses_postprocessed("enron_responses_postprocessed")
    create_table_responses_postprocessed("enron_responses_postprocessed_101_201")
    create_table_responses_postprocessed("bbc_responses_postprocessed")
    create_table_responses_postprocessed("bbc_responses_postprocessed2")

    delete_table("enron_text_all")
    merge_tables_into_new("enron_text2", "enron_text3", "enron_text_all")

    update_summary_question("enron_responses")
    update_summary_question("enron_responses2")
    update_summary_question("enron_responses3")
    update_summary_question("bbc_responses2")
    update_summary_question("enron_responses_postprocessed")
    update_summary_question("enron_responses_postprocessed_101_201")
    update_summary_question("bbc_responses_postprocessed")

    merge_response_tables_into_new("enron_responses2", "enron_responses3", "enron_responses_all")
    
    merge_postprocessed_tables_into_new("enron_responses_postprocessed", "enron_responses_postprocessed_101_201", "enron_responses_postprocessed_all")
    merge_postprocessed_tables_into_new("bbc_responses_postprocessed", "bbc_responses_postprocessed2", "bbc_responses_postprocessed_all")
    """