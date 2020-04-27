"""Import local directory for local imports"""

# Standard imports
from os.path import abspath, dirname
import sys

sys.path.insert(0, dirname(abspath(__file__)))
