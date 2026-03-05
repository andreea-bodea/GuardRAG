#!/usr/bin/env python3
"""
Database Backup Script for EACL_RAG Project

This script creates a comprehensive backup of all database tables by:
1. Creating a timestamped backup directory
2. Exporting all tables to CSV files
3. Generating a backup manifest with metadata
4. Creating a compressed archive of the backup

Usage:
    python backup_database.py [--output-dir BACKUP_DIR] [--compress]
"""

import os
import sys
import csv
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# List of all tables in the database (from the image description)
ALL_TABLES = [
    'bbc_responses2',
    'bbc_responses_postprocessed',
    'bbc_responses_postprocessed2',
    'bbc_responses_postprocessed_all',
    'bbc_text',
    'bbc_text2',
    'enron_responses',
    'enron_responses2',
    'enron_responses3',
    'enron_responses_all',
    'enron_responses_postprocessed',
    'enron_responses_postprocessed_101_201',
    'enron_responses_postprocessed_all',
    'enron_text',
    'enron_text2',
    'enron_text3',
    'enron_text_all'
]

def get_table_info(table_name):
    """Get basic information about a table (row count, columns, etc.)"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get row count
            cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = cur.fetchone()['count']
            
            # Get column information
            cur.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cur.fetchall()
            
            return {
                'row_count': row_count,
                'columns': [dict(col) for col in columns]
            }
    except Exception as e:
        print(f"Error getting info for table {table_name}: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def export_table_to_csv(table_name, output_file):
    """Export a single table to CSV format"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH CSV HEADER", f)
        return True
    except Exception as e:
        print(f"Error exporting table '{table_name}': {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def create_backup_manifest(backup_dir, table_info):
    """Create a manifest file with backup metadata"""
    manifest = {
        'backup_timestamp': datetime.now().isoformat(),
        'database_url': DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost',  # Hide credentials
        'total_tables': len(ALL_TABLES),
        'tables': table_info,
        'total_records': sum(info['row_count'] for info in table_info.values() if info),
        'backup_version': '1.0'
    }
    
    manifest_file = os.path.join(backup_dir, 'backup_manifest.json')
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest_file

def create_compressed_backup(backup_dir, compress=True):
    """Create a compressed tar.gz archive of the backup"""
    if not compress:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"database_backup_{timestamp}.tar.gz"
    archive_path = os.path.join(os.path.dirname(backup_dir), archive_name)
    
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(backup_dir, arcname=os.path.basename(backup_dir))
    
    return archive_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a comprehensive backup of all database tables')
    parser.add_argument('--output-dir', default=None, help='Output directory for backup (default: ./backups/backup_YYYYMMDD_HHMMSS)')
    parser.add_argument('--compress', action='store_true', help='Create compressed tar.gz archive')
    parser.add_argument('--tables', nargs='*', help='Specific tables to backup (default: all tables)')
    
    args = parser.parse_args()
    
    # Determine tables to backup
    tables_to_backup = args.tables if args.tables else ALL_TABLES
    
    # Create backup directory
    if args.output_dir:
        backup_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"./backups/backup_{timestamp}"
    
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Creating backup in: {backup_dir}")
    
    # Get table information and export tables
    table_info = {}
    successful_exports = 0
    failed_exports = 0
    
    for table_name in tables_to_backup:
        print(f"Processing table: {table_name}")
        
        # Get table info
        info = get_table_info(table_name)
        if info:
            table_info[table_name] = info
            print(f"  - Rows: {info['row_count']}")
            print(f"  - Columns: {len(info['columns'])}")
        
        # Export to CSV
        csv_file = os.path.join(backup_dir, f"{table_name}.csv")
        if export_table_to_csv(table_name, csv_file):
            successful_exports += 1
            print(f"  ✓ Exported to {csv_file}")
        else:
            failed_exports += 1
            print(f"  ✗ Failed to export {table_name}")
    
    # Create manifest
    manifest_file = create_backup_manifest(backup_dir, table_info)
    print(f"Created manifest: {manifest_file}")
    
    # Create compressed archive if requested
    archive_path = None
    if args.compress:
        print("Creating compressed archive...")
        archive_path = create_compressed_backup(backup_dir, compress=True)
        if archive_path:
            print(f"Created archive: {archive_path}")
            # Optionally remove the uncompressed backup directory
            # shutil.rmtree(backup_dir)
    
    # Summary
    print("\n" + "="*50)
    print("BACKUP SUMMARY")
    print("="*50)
    print(f"Backup directory: {backup_dir}")
    print(f"Tables processed: {len(tables_to_backup)}")
    print(f"Successful exports: {successful_exports}")
    print(f"Failed exports: {failed_exports}")
    print(f"Total records backed up: {sum(info['row_count'] for info in table_info.values() if info)}")
    if archive_path:
        print(f"Compressed archive: {archive_path}")
    print("="*50)
    
    if failed_exports > 0:
        print(f"WARNING: {failed_exports} tables failed to export. Check the error messages above.")
        return 1
    else:
        print("✓ All tables backed up successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
