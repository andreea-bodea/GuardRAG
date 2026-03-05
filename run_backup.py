#!/usr/bin/env python3
"""
Simple script to run the database backup
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Data.backup_database import main

if __name__ == "__main__":
    # Run the backup with default settings
    # This will create a timestamped backup directory with all tables
    exit_code = main()
    sys.exit(exit_code)
