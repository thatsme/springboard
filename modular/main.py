
#!/usr/bin/env python
#make executable in bash chmod +x PyRun
import warnings 
warnings.filterwarnings('ignore')

import sys
import inspect
import importlib
import os
import pandas as pd
print("Pandas => ", pd.__version__)
from MlUtil import MlUtil

if __name__ == "__main__":
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)

    #else:
    #    print("--> ",cmd_folder)


    train_dataset = pd.read_csv('../data/titanic_train.csv')
    test_dataset = pd.read_csv('../data/titanic_test.csv')


    m = MlUtil(train_dataset, test_dataset, vb=3)
    print("MlUtil => ", m.__version__)

    print(train_dataset.head())

    #m.transform_title_new("Title", "Name")
    #m.transform_dummies(["Title"], ["Title"], True)
