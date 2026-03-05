"""
Helper script to prepare TAB dataset for processing.
This script helps verify and prepare your TAB dataset CSV file.
"""
import csv
import os
import sys

def inspect_csv(file_path, num_rows=5):
    """
    Inspect CSV file to understand its structure.
    
    Args:
        file_path: Path to CSV file
        num_rows: Number of rows to display
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Inspecting CSV file: {file_path}")
    print(f"{'='*60}\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Read header
        try:
            header = next(reader)
            print(f"Header ({len(header)} columns):")
            for i, col in enumerate(header):
                print(f"  Column {i}: {col}")
            print()
        except StopIteration:
            print("Warning: CSV file appears to be empty")
            return
        
        # Read sample rows
        print(f"Sample rows (first {num_rows}):")
        for row_num, row in enumerate(reader, start=1):
            if row_num > num_rows:
                break
            
            print(f"\nRow {row_num}:")
            for i, cell in enumerate(row):
                cell_preview = cell[:100] + "..." if len(cell) > 100 else cell
                print(f"  Column {i}: {cell_preview}")
    
    print(f"\n{'='*60}\n")

def prepare_tab_csv(input_file, output_file, text_column=0, max_length=None):
    """
    Prepare TAB dataset CSV for processing.
    Extracts text column and optionally filters by length.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        text_column: Index of column containing text (default: 0)
        max_length: Maximum text length in characters (None = no limit)
    """
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    print(f"\nPreparing TAB dataset...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Text column: {text_column}")
    if max_length:
        print(f"Max length: {max_length} characters")
    print()
    
    processed = 0
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['text'])
        
        # Process rows
        for row_num, row in enumerate(reader, start=1):
            if len(row) <= text_column:
                print(f"Warning: Row {row_num} doesn't have column {text_column}, skipping")
                skipped += 1
                continue
            
            text = row[text_column].strip()
            
            if not text:
                skipped += 1
                continue
            
            if max_length and len(text) > max_length:
                print(f"Row {row_num}: Text too long ({len(text)} chars), skipping")
                skipped += 1
                continue
            
            writer.writerow([text])
            processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed} rows...")
    
    print(f"\n{'='*60}")
    print(f"Preparation complete!")
    print(f"Processed: {processed} rows")
    print(f"Skipped: {skipped} rows")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare TAB dataset for processing')
    parser.add_argument('action', choices=['inspect', 'prepare'], 
                       help='Action to perform: inspect or prepare')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='TAB_preprocessed.csv',
                       help='Output CSV file path (for prepare action)')
    parser.add_argument('--column', '-c', type=int, default=0,
                       help='Column index containing text (default: 0)')
    parser.add_argument('--max-length', '-m', type=int, default=None,
                       help='Maximum text length in characters (default: no limit)')
    parser.add_argument('--rows', '-r', type=int, default=5,
                       help='Number of rows to display (for inspect action)')
    
    args = parser.parse_args()
    
    if args.action == 'inspect':
        inspect_csv(args.input_file, args.rows)
    elif args.action == 'prepare':
        prepare_tab_csv(args.input_file, args.output, args.column, args.max_length)

