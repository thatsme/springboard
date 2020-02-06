
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
from Estimator import Estimator

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

    m.transform_title_new("Title", "Name")
    m.transform_dummies(["Title"], ["Title"], True)

    #p = m.getCombined()
    #print(p.Title.head())

    m.process_age()
    m.transform_age_new()

    m.transform_age_group("AgeGroup", "Age")
    m.transform_dummies(["AgeGroup"], ["AG"], True)

    m.transform_surname_new()

    m.transform_lenght_new(["NameLength"], ["Name"])

    m.transform_surname_members_new()

    m.transform_cabintype_new()

    m.transform_hascabin_new()

    m.transform_splitcabin_new()

    m.transform_famsize_new()

    m.transform_isalone_new()

    m.transform_person_new()

    m.transform_socialstatus(["Social"], ["Pclass"])

    m.transform_dummies(["Social"], ["SC"], 1)

    m.handling_missings("Embarked", "value", "S")

    m.handling_missings("Fare", "median")

    m.transform_dummies(["Cabin_Deck", "Person", "Sex", "Embarked"], ["Deck", "P", "S", "Embarked"], True)

    m.transform_int(["Fare", "Age", "CabinType"], "int")

    # Fare x Person transformation ( need to check null values in Fare column on test data before)
    m.transform_farexperson_new()

    # Drop unecessary columns
    m.transform_drop(["Title", "Embarked", "Person", "Name", "Cabin", "Cabin_Deck", "Cabin_Number", "Sex", "Surname", "Ticket", "Age", "AgeGroup", "Social"])

    m.transform_drop(["Pclass", "SibSp"])

    m.df_combined.head()



    ## Define my feature list 
    ##
    #myList = ["Survived", "female", "male", "Age", "male_adult", "female_adult", "child", "TitleCat", "Pclass", "NameLength", "CabinType", "CabinCat", "SibSp", "Parch", "Fare", "Embarked", "Surname_Members", "Ticket_Members", "Fam_Size"]
    #myList = ["female","male","male_adult","female_adult","child","TitleCat","Pclass","NameLength","CabinType","CabinCat","Fare","Embarked", "Cabin_Deck"]

    myList = ["Survived", "female", "male_adult", "child", "TitleCat", "Pclass", "CabinCat", "SibSp", "Fare", "Embarked", "Surname_Members", "Fam_Size"]
    myListClean = ["female", "male_adult", "child", "TitleCat", "Pclass", "CabinCat", "SibSp", "Fare", "Embarked", "Surname_Members", "Fam_Size"]

    ##myList = ["female","male","male_adult","female_adult","child", "Age", "IsAlone", "TitleCat","Pclass","NameLength","Fare", "Cabin_Deck", "Embarked"]

    myList1 = ["male_adult","female_adult","TitleCat","Pclass","NameLength","Fare", "Cabin_Deck"]

    myListPlusSurvived = ["Survived", "female","male","male_adult","female_adult","child","TitleCat","Pclass","NameLength","Fare","Embarked", "Cabin_Deck"]

    #myTitan.set_features(myListClean)

    erf = Estimator(RandomForestClassifier(),"RandomForestClassifier")

    m.inject_estimator(erf)

    #tr = myTitan.train_data[myTitan.features]
    tr = m.train_data
    #myTitan.display_all(tr)
    #ts = myTitan.test_data[myTitan.features]
    m.RFE(tr, 13)

    m.LASSO(tr)


    cc = m.train_data.columns.values

    ccl = list(cc)
    ccp = list(cc)
    ccl.remove("Survived")
    ccl.remove("PassengerId")
    ccl.remove("Fare")
    ccl.remove("Has_Cabin")
    ccl.remove("Deck_T")
    ccl.remove("Deck_G")
    ccl.remove("Deck_F")
    ccl.remove("Deck_D")
    ccl.remove("Deck_B")
    ccl.remove("Title_Officier")
    ccl.remove("Title_Royalty")
    ccl.remove("Embarked_Q")
    ccl.remove("Parch")
    ccl.remove("IsAlone")
    ccl.remove("Singleton")
    ##ccl.remove("Deck_C")
    ##ccl.remove("Deck_A")
    #ccl.remove("Embarked_C")
    #ccl.remove("Deck_A")
    #ccl.remove("Deck_B")
    #ccl.remove("Deck_C")
    #ccl.remove("Deck_D")
    #ccl.remove("Deck_E")
    #ccl.remove("Deck_F")
    #ccl.remove("Deck_G")
    #ccl.remove("Singleton")

    bha = ['Fare','Title_Mr', 'NameLength', 'Surname_Members','CabinType', 'Fam_Size', 'Deck_Z', 'P_female_adult', 'P_male_adult', 'S_male','Fare_Per_Person']

    m.set_features(bha)

    m.features

    m.config.read('..\\data\\MlUtil.ini')
    m.modelTestRandomRun = 0

    # Start model testing
    m.modelRandomTest()
