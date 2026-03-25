"""
Simple runner script to execute the analysis
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import main
if __name__ == "__main__":
    main()