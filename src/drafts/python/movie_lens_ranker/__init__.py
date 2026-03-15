import os
import sys
current_dir = os.getcwd()
two_levels_up = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(two_levels_up + "/src/main/python")
