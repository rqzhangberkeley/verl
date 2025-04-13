#!/usr/bin/env python3
"""
Debug wrapper for main_ppo.py
"""
import os
import sys

# Ensure the correct Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the main function
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    # Call the main function with the same arguments
    main() 