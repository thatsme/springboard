
#!/usr/bin/env python
#make executable in bash chmod +x PyRun

import sys
import inspect
import importlib
import os
import pandas as pd

if __name__ == "__main__":
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)

    else:
        print("--> ",cmd_folder)


    train_dataset = pd.read_csv('../data/titanic_train.csv')
    test_dataset = pd.read_csv('../data/titanic_test.csv')
