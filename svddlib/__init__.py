# svddlib/__init__.py

from .svdd import SVDD
#from .visualizer import DecisionBoundaryVisualizer 
# #--> from svddlib import SVDD, DecisionBoundaryVisualizer instead of 
# --> from svddlib.svdd import SVDD,  from svddlib.visualizer import DecisionBoundaryVisualizer



__all__ = ["SVDD", "DecisionBoundaryVisualizer"]
