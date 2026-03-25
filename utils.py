"""
Utility functions for motor imagery BCI analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)