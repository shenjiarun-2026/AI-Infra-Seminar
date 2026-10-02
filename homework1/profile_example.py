"""Compatibility entry point for the three Transformers Attention traces.

Use: python homework1/profile_example.py --gpu 4
The profiling scope itself is in run.py::profile_forward().
"""

from run import main

if __name__ == "__main__":
    main()
