import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

from transformers import (CategoriesExtractor, CountryTransformer, GoalAdjustor,
                          TimeTransformer)
                          
class MlUtil:
# class that implements titanic EDA and Model

    # constructor
    def __init__(self, train=None, test=None, vb=0, model=None):
        self.verbose = vb
        self.__version__ = "1.2"
        if(train is not None and test is not None):    
            self.train_data = train
            self.test_data = test
            # Combine the two datasets
            self.df_combined = self.train_data.append(self.test_data)
            self.combine = [self.train_data, self.test_data]
            self.wichname = ["Training data", "Testing data"]
            self.y = self.train_data['Survived'].ravel()

            # Some useful parameters which will come in handy later on
            self.ntrain = self.train_data.shape[0]
            self.ntest = self.test_data.shape[0]
            self.ncombined = self.df_combined.shape[0]
            self.SEED = 0 # for reproducibility
            self.NFOLDS = 5 # set folds for out-of-fold prediction
            #self.kf = KFold(self.ntrain, n_folds= self.NFOLDS, random_state=self.SEED)
            # sklearn >= 0.20
            self.kf = KFold(n_splits=self.NFOLDS, random_state=self.SEED)
        self.Title_Dictionary = {
            "Capt" : "Officier",
            "Col" : "Officier", 
            "Major" : "Officier",
            "Jonkheer" : "Royalty",
            "Don" : "Royalty",
            "Sir" : "Royalty",
            "Dr" : "Officier",
            "Rev" : "Officier",
            "the Countess" : "Royalty",
            "Mme" : "Mrs",
            "Mlle" : "Miss",
            "Ms" : "Mrs",
            "Mr" : "Mr",
            "Mrs" : "Mrs",
            "Miss" : "Miss",
            "Master" : "Master",
            "Lady" : "Royalty"
        }
        
    def display_all(self, df):
        with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
            display(df)
            
    def RFE(self, train, feat):
        y = train["Survived"]
        X = train.drop("Survived",1)  
        #X_train, y_train, X_test, y_test = train_test_split(train, y, test_size=0.2)
        
        model = LinearRegression()
        #Initializing RFE model
        rfe = RFE(model, feat)
        #Transforming data using RFE
        X_rfe = rfe.fit_transform(X, y)  
        #Fitting the data to model
        model.fit(X_rfe, y)
        print(rfe.support_)
        print(rfe.ranking_)
        
    def LASSO(self, train):
        y = train["Survived"]
        X = train.drop("Survived",1)  
        reg = LassoCV()
        reg.fit(X, y)
        print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        print("Best score using built-in LassoCV: %f" %reg.score(X,y))
        coef = pd.Series(reg.coef_, index = X.columns)
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
        imp_coef = coef.sort_values()
        import matplotlib
        matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        imp_coef.plot(kind = "barh")
        plt.title("Feature importance using Lasso Model")
        #print(coef)
        
    def RIDGE(self, train):
        y = train["Survived"]
        X = train.drop("Survived",1)  
        alphas = 10**np.linspace(10,-2,100)*0.5
        ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
        ridgecv.fit(X, y)
        print(ridgecv.alpha_)
        
    
    def get_transform_methods(self):
        method_list = [func for func in dir(Titanic) if callable(getattr(Titanic, func)) and func.startswith("transform_")]
        print(method_list)
        
    def combine_datasets(self):
        self.combine = [self.train_data, self.test_data]
        
    def head(self, n=5):
        for datasets, names in zip(self.combine, self.wichname):
            print(datasets.head(n))
    
    def tail(self, n=5):
        for datasets, names in zip(self.combine, self.wichname):
            print(datasets.tail(n))
        
    def info(self):
        for datasets, names in zip(self.combine, self.wichname):
            print("Information for dataset : "+names)
            print(datasets.info())

    def describe(self):
        for datasets, names in zip(self.combine, self.wichname):
            print("Describe for dataset : "+names)
            print(datasets.describe())
            
    def setTarget(self, column):
        self.train_data = self.df_combined[:self.ntrain].copy()
        self.test_data = self.df_combined[self.ntrain:].copy()
        self.target = self.train_data[column].values
        self.test_data.drop(column, axis=1, inplace=True)
    
    def getTest(self):
        return self.test_data[self.features]

    def getTrain(self):
        return self.train_data[self.features]

    def survived_stats(self):
        # Survived by cabin class
        self.train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
        # Survived by Sex
        self.train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
        # Survived by number of Parent ( seams with no correlations ) 
        self.train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
        # Survived by number of Parent ( seams with no correlations ) 
        self.train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
        
    # function to get oven/odd/null from cabine
    def get_type_cabine(self, cabine):
        rt = {
            "even" : 2
            "odd" : 1
            "none" : 0
        }

        # Use a regular expression to search for a number.
        cabine_search = re.search("\d+", cabine)
        # If the number exists, extract and return it.
        if cabine_search:
            num = cabine_search.group(0)
            if np.float64(num) % 2 == 0:
                return tr.even
            else:
                return rt.odd
        return rt.none

    def get_evenodd(self, value):
        # Use a regular expression to search for a number.
        number_search = re.search("\d+", value)
        # If the number exists, extract and return it.
        if value_search:
            num = value_search.group(0)
            if np.float64(num) % 2 == 0:
                return "2"
            else:
                return "1"
        return "0"

    # % of survived woman
    def woman_survived(self):
        women = self.train_data.loc[self.train_data.Sex == 'female']["Survived"]
        rate_women = sum(women)/len(women)
        print("% of women who survived:", rate_women)
        
    # % of survived man
    def man_survived(self):
        man = self.train_data.loc[self.train_data.Sex == 'male']["Survived"]
        rate_man = sum(man)/len(man)
        print("% of man who survived:", rate_man)
    
    # Output of survived by cabin/class
    def cabinclass_survived(self):
        # Survived by cabin class
        self.train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # Output of survived by sibsp
    def sibsp_survived(self):
        # Survived by number of Parent ( seams with no correlations ) 
        self.train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    
    # Output of survived by Parch    
    def parch_survived(self):
        # Survived by number of Parent ( seams with no correlations ) 
        self.train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
        
    ####
    #### Handling missing values ( implements value/median/mean/std/max/min)
    ####
    def handling_missings(self, column, function, value=None):    
        if(function=="value"):
            self.df_combined[column] = self.df_combined[column].fillna(value)
        elif(function=="median"):
            value = self.df_combined[column].median()
            self.df_combined[column] = self.df_combined[column].fillna(value)
        elif(function=="mean"):
            value = self.df_combined[column].median()
            self.df_combined[column] = self.df_combined[column].fillna(value)
        elif(function=="std"):
            value = self.df_combined[column].std()
            self.df_combined[column] = self.df_combined[column].fillna(value)
        elif(function=="max"):
            value = self.df_combined[column].max()
            self.df_combined[column] = self.df_combined[column].fillna(value)
        elif(function=="min"):
            value = self.df_combined[column].min()
            self.df_combined[column] = self.df_combined[column].fillna(value)
    
    ####
    #### TRANSFORMATIONS
    ####
    # Get the lenght of column and save it in a newcolumn
    def transform_lenght(self,newcolumn, column):
        for nc, c in zip(newcolumn, column):
            self.df_combined[nc] = self.df_combined[c].apply(lambda x: len(x))
        if(self.verbose>1):
            print("Transformation done")
    
    # Transform the column in an integer
    def transform_int(self, column, function):
        for c in column:
            if(function=="int"):
                self.df_combined[c] = self.df_combined[c].astype(int)
        if(self.verbose>1):
            print("Transformation done")

    # Transform the column in an float
    def transform_int(self, column, function):
        for c in column:
            if(function=="float"):
                self.df_combined[c] = self.df_combined[c].astype(float)
        if(self.verbose>1):
            print("Transformation done")

    # Transform column in dummies        
    def transform_dummies(self, column, prefix, axis=1):
        for c, p in zip(column,prefix):
            self.df_combined = pd.concat([self.df_combined, pd.get_dummies(self.df_combined[c], prefix=p)], axis=axis)
        if(self.verbose>1):
            print("Transformation done")
    
    def transform_drop(self, column, inplace=True):
        for c in column:
            self.df_combined.drop(c, axis=1, inplace=inplace)

        if(self.verbose>1):
            print("Transformation done")
            
    # Transform title ( Extract title from colum -> Name, and set newcolumn)
    def transform_title_new(self, newcolumn, column):
        # Transformations on Name column
        self.df_combined[newcolumn] = self.df_combined[column].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))            
        self.df_combined[newcolumn] = self.df_combined[newcolumn].map(self.Title_Dictionary)
            
        if(self.verbose>1):
            print("Transformation done")
            
    # Create category for title column
    def transform_titlecat(self, newcolumn, column, mappings=None):
        if(mappings is not None):
            title_mapping = mappings
        else:
            title_mapping = {
                "Mr": 1,
                "Miss": 2,
                "Mrs": 3,
                "Master": 4,
                "Dr": 5,
                "Rev": 6,
                "Major": 7,
                "Col": 7,
                "Mlle": 2,
                "Mme": 3,
                "Don": 9,
                "Dona": 9,
                "Lady": 10,
                "Countess": 10,
                "Jonkheer": 10,
                "Sir": 9,
                "Capt": 7,
                "Ms": 2,
            }
        for datasets, names in zip(self.combine, self.wichname):
            datasets[newcolumn] = datasets[column].map(title_mapping)
        if(self.verbose>1):
            print("Transformation done")
            
    # Extract surname
    def transform_surname_new(self):
        self.df_combined["Surname"] = self.df_combined["Name"].apply(lambda x: x.split(",")[0].lower())
        if(self.verbose>1):
            print("Transformation done")

    def transform_surname_members_new(self):
        table_ticket = pd.DataFrame(self.df_combined["Surname"].value_counts())
        table_ticket.rename(columns={"Surname": "Surname_Members"}, inplace=True)
        self.df_combined = pd.merge(
                self.df_combined,
                table_ticket,
                left_on="Surname",
                right_index=True,
                how="left",
                sort=False,
        )
        if(self.verbose>1):
            print("Transformation done")


    def transform_cabintype_new(self):
        self.df_combined["Cabin"] = self.df_combined["Cabin"].fillna(" ")
        self.df_combined["CabinType"] = self.df_combined["Cabin"].apply(self.get_type_cabine)
        if(self.verbose>1):
            print("Transformation done")

        
    def transform_cabincat(self):
        for datasets, names in zip(self.combine, self.wichname):
            datasets["CabinCat"] = pd.Categorical(datasets["Cabin"].fillna("0").apply(lambda x: x[0])).codes    
        if(self.verbose>1):
            print("Transformation done")
            
    def transform_hascabin_new(self):
        # New Column Has_Cabin
        self.df_combined['Has_Cabin'] = ~self.df_combined.Cabin.isnull()
        if(self.verbose>1):
            print("Transformation done")
            
    def transform_splitcabin_new(self):
        # New Column Cabin_Number, Cabin_Deck
        self.df_combined['Cabin_Number'] = self.df_combined['Cabin'].str.replace('([A-Z]+)', '')
        self.df_combined['Cabin_Deck'] = self.df_combined['Cabin'].str.extract('([A-Z]+)')
        self.df_combined['Cabin_Deck'] = self.df_combined['Cabin_Deck'].fillna('Z')
        if(self.verbose>1):
            print("Transformation done")
                        
    def transform_cabindeck_new(self):
        self.df_combined["Cabin_Deck"] = pd.Categorical(self.df_combined["Cabin_Deck"]).codes
        if(self.verbose>1):
            print("Transformation done")
            
    def transform_removedeck(self, deck):
        self.train_data = self.train_data[self.train_data['Cabin_Deck'] != deck]
        self.test_data = self.test_data[self.test_data['Cabin_Deck'] != deck]
        self.combine = [self.train_data, self.test_data]
        if(self.verbose>1):
            print("Transformation done")
        
    def transform_famsize_new(self):
        # New Column Fam_Size 
        self.df_combined['Fam_Size'] = self.df_combined["Parch"] + self.df_combined["SibSp"] +1 
        self.df_combined['Singleton'] = self.df_combined['Fam_Size'].map(lambda s: 1 if s == 1 else 0)
        self.df_combined['SmallFamily'] = self.df_combined['Fam_Size'].map(lambda s:1 if 2 <= s <= 4 else 0)
        self.df_combined['LargeFamily'] = self.df_combined['Fam_Size'].map(lambda s:1 if 5 <= s else 0)
        if(self.verbose>1):
            print("Transformation done")

    def transform_isalone_new(self):
        # New Column Fam_Size 
        self.df_combined['IsAlone'] = 0
        self.df_combined.loc[self.df_combined['Fam_Size'] == 1, 'IsAlone'] = 1
            
        if(self.verbose>1):
            print("Transformation done")
            
    def transform_farexperson_new(self):
        # Fare per person
        self.df_combined['Fare_Per_Person'] = self.df_combined['Fare']/(self.df_combined['Fam_Size']+1)
        self.df_combined['Fare_Per_Person'] = self.df_combined['Fare_Per_Person'].astype(int)
        if(self.verbose>1):
            print("Transformation done")
            
                            
    def transform_quant_age(self, n=4, dpl='raise'):
        # Quantization of Age for datasets in 8 segments (default = 4)
        for datasets, names in zip(self.combine, self.wichname):
            datasets['Age'] = datasets['Age'].astype(int)
            datasets['CatAge'] = pd.qcut(datasets["Age"], q=n, labels=False, duplicates=dpl )
        if(self.verbose>1):
            print("Transformation done")
            print (self.train_data[['CatAge', 'Survived']].groupby(['CatAge'], as_index=False).mean())
                  
    def transform_age(self, n=4):
        # Mapping Age
        for datasets, names in zip(self.combine, self.wichname):
            datasets.loc[ datasets['Age'] <= 16, 'Age'] 					       = 0
            datasets.loc[(datasets['Age'] > 16) & (datasets['Age'] <= 32), 'Age'] = 1
            datasets.loc[(datasets['Age'] > 32) & (datasets['Age'] <= 48), 'Age'] = 2
            datasets.loc[(datasets['Age'] > 48) & (datasets['Age'] <= 64), 'Age'] = 3
            datasets.loc[ datasets['Age'] > 64, 'Age']                        
        if(self.verbose>1):
                print("Transformation done")
        
    def transform_fare(self, n=4):
        # Mapping Fare
        for datasets, names in zip(self.combine, self.wichname):
            # Mapping Fare
            datasets.loc[ datasets['Fare'] <= 7.91, 'Fare'] 						        = 0
            datasets.loc[(datasets['Fare'] > 7.91) & (datasets['Fare'] <= 14.454), 'Fare'] = 1
            datasets.loc[(datasets['Fare'] > 14.454) & (datasets['Fare'] <= 31), 'Fare']   = 2
            datasets.loc[ datasets['Fare'] > 31, 'Fare'] 							        = 3
        if(self.verbose>1):
                print("Transformation done")
                
    def transform_quant_fare(self, n = 8, dpl = 'raise'):
        # Quantization of Fare for datasets in 8 segments ( defaul = 4)
        for datasets, names in zip(self.combine, self.wichname):
            datasets['Fare'] = datasets['Fare'].astype(int)
            datasets['CatFare']= pd.qcut(datasets["Fare"], q=n, labels=False, duplicates=dpl)
        if(self.verbose>1):
            print("Transformation done ")
            print (self.train_data[['CatFare', 'Survived']].groupby(['CatFare'], as_index=False).mean())
        
    def transform_ageclass(self):
        for datasets, names in zip(self.combine, self.wichname):     
            datasets['Age_Class'] = datasets['Age']* datasets['Pclass']
            datasets['Age_Class'] = datasets['Age_Class'].astype(int)
        if(self.verbose>1):
            print("Transformation done")
    
    def transform_embarked(self):
        for datasets, names in zip(self.combine, self.wichname):     
            datasets["Embarked"] = pd.Categorical(datasets["Embarked"]).codes
        if(self.verbose>1):
            print("Transformation done")
        
    def get_person(self, passenger):
        child_age = 18
        age, sex = passenger
        if age < child_age:
            return "child"
        elif sex == "female":
            return "female_adult"
        else:
            return "male_adult"
    
    def transform_person_new(self, getdummies=False):
        self.df_combined['Person'] = self.df_combined[["Age", "Sex"]].apply(self.get_person, axis=1)
        if(self.verbose>1):
            print("Transformation done")

        
    def transform_ticket_members(self):
        for datasets, names in zip(self.combine, self.wichname):
            table_ticket = pd.DataFrame(datasets["Ticket"].value_counts())
            table_ticket.rename(columns={"Ticket": "Ticket_Members"}, inplace=True)
            if(names=="Training data"):
                self.train_data = pd.merge(
                    self.train_data,
                    table_ticket,
                    left_on="Ticket",
                    right_index=True,
                    how="left",
                    sort=False,
                )
            else:
                self.test_data = pd.merge(
                    self.test_data,
                    table_ticket,
                    left_on="Ticket",
                    right_index=True,
                    how="left",
                    sort=False,
                )
                
        self.combine = [self.train_data, self.test_data]
        if(self.verbose>1):
            print("Transformation done")

        
    def process_age(self):
        self.grouped_train = self.df_combined.groupby(['Sex', 'Pclass', 'Title'])
        self.grouped_median_train = self.grouped_train.median()
        self.grouped_median_train = self.grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', "Age"]]
        
    def fill_age(self, row):
        condition = (
            (self.grouped_median_train['Sex'] == row['Sex']) &
            (self.grouped_median_train['Title'] == row['Title']) &
            (self.grouped_median_train['Pclass'] == row['Pclass'])
        )
        if np.isnan(self.grouped_median_train[condition]['Age'].values[0]):
            print('true')
            condition = (
                (self.grouped_median_train['Sex'] == row['Sex']) &
                (self.grouped_median_train['Pclass'] == row['Pclass'])
            )
        return self.grouped_median_train[condition]['Age'].values[0]
    
    def transform_age_new(self):
        self.df_combined['Age'] = self.df_combined.apply(lambda row: self.fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
        if(self.verbose>1):
            print("Transformation done")

    def transform_guess_age(self, myClasser=None):
        if(myClasser == None):
            classers = [
                "Fare",
                "Parch",
                "Pclass",
                "SibSp",
                "TitleCat",
                "CabinCat",
                "female",
                "male",
                "Embarked",
                "Fam_Size",
                "NameLength",
                "Ticket_Members"
                #"Ticket_Id",
            ]
        else:
            classer = myClasser
            
        etr = ExtraTreesRegressor(n_estimators=200)
        for datasets, names in zip(self.combine, self.wichname):
            X_train = datasets[classers][datasets["Age"].notnull()]
            Y_train = datasets["Age"][datasets["Age"].notnull()]

            X_test = datasets[classers][datasets["Age"].isnull()]
            etr.fit(X_train, np.ravel(Y_train))
            age_preds = etr.predict(X_test)

            datasets["Age"][datasets["Age"].isnull()] = age_preds
        if(self.verbose>1):
            print("Transformation done")
        
    ####
    #### Display section
    ####
    
    def display_title(self, column = None):
        xvalue = 'Title'
        xloc = 'upper left'
        if(column is not None):
            xvalue = column
            
        for datasets, names in zip(self.combine, self.wichname):
            g = sns.countplot(x=xvalue, data=datasets);
            plt.legend(title='Title from Name column for '+names, loc=xloc)
            plt.xticks(rotation=45);
            plt.show(g)

    def display_hascabin(self, column = None):
        xvalue = "Has_Cabin"
        xloc = 'upper left'
        if(column is not None):
            xvalue = column
            
        for datasets, names in zip(self.combine, self.wichname):
            g = sns.countplot(x=xvalue, data=datasets);
            plt.legend(title='Has Cabin for '+names, loc=xloc)
            plt.xticks(rotation=45);
            plt.show(g)

    def display_cabindeck(self, column = None):
        xvalue = 'Cabin_Deck'
        xloc = 'upper left'
        if(column is not None):
            xvalue = column
        
        for datasets, names in zip(self.combine, self.wichname):
            g = sns.countplot(x=xvalue, data=datasets);
            plt.legend(title='Cabin Deck for '+names, loc=xloc)
            plt.xticks(rotation=45);
            plt.show(g)
            
    def display_famsize(self, column = None):
        xvalue = 'Fam_Size'
        xloc = 'upper left'
        if(column is not None):
            xvalue = column
        
        for datasets, names in zip(self.combine, self.wichname):
            g = sns.countplot(x=xvalue, data=datasets);
            plt.legend(title='Family Size Survived for '+names, loc=xloc)
            plt.xticks(rotation=45);
            plt.show(g)

    def display_farexperson(self, column = None):
        xvalue = 'Fare_Per_Person'
        xloc = 'upper left'
        if(column is not None):
            xvalue = column
        
        for datasets, names in zip(self.combine, self.wichname):
            g = sns.countplot(x=xvalue, data=datasets);
            plt.legend(title='Fare per Person for '+names, loc=xloc)
            plt.xticks(rotation=45);
            plt.show(g)

    def display_quant_age(self, column = None):
        xvalue = 'CatAge'
        xloc = 'upper left'
        if(column is not None):
            xvalue = column

        for datasets, names in zip(self.combine, self.wichname):
            g = sns.countplot(x=xvalue, data=datasets);
            plt.legend(title='Category of Ages for '+names, loc=xloc)
            plt.xticks(rotation=45);
            plt.show(g)

    def display_quant_fare(self, column = None):
        xvalue = 'CatFare'
        xloc = 'upper left'
        if(column is not None):
            xvalue = column
            
        for datasets, names in zip(self.combine, self.wichname):
            g = sns.countplot(x=xvalue, data=datasets);
            plt.legend(title='Category of Fares for '+names, loc=xloc)
            plt.xticks(rotation=45);
            plt.show(g)        
         
    def display_pearson(self, target, limit = 0.5):
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14,12))
        cor = self.train_data[self.features].astype(float).corr()
        #cor = self.train_data.astype(float).corr()
        #Correlation with output variable
        cor_target = abs(cor[target])
        #Selecting highly correlated features
        relevant_features = cor_target[cor_target>limit]
        print(relevant_features)
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(cor,linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
        
    def display_pairplots(self):
        g = sns.pairplot(self.train_data[self.features], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
        g.set(xticklabels=[])
        
    def display_rf_importance(self):
        rfd = Display_Importance(self.feature_dataframe, 'Random Forest feature importances', 'features')        
        rfd.plot()

    def display_et_importance(self):
        etd = Display_Importance(self.feature_dataframe, 'Extra Trees  feature importances', 'features')
        etd.plot()

    def display_ada_importance(self):
        adad = Display_Importance(self.feature_dataframe, 'AdaBoost feature importances', 'features')
        adad.plot()

    def display_gb_importance(self):
        gbd = Display_Importance(self.feature_dataframe, 'Gradient Boost feature importances', 'features')
        gbd.plot()
        
    def display_mean_feature_importance(self):
        y = self.feature_dataframe_mean['mean'].values
        x = self.feature_dataframe_mean['features'].values
        data = [go.Bar(
                    x= x,
                     y= y,
                    width = 0.5,
                    marker=dict(
                       color = self.feature_dataframe_mean['mean'].values,
                    colorscale='Portland',
                    showscale=True,
                    reversescale = False
                    ),
                    opacity=0.6
                )]

        layout= go.Layout(
            autosize= True,
            title= 'Barplots of Mean Feature Importance',
            hovermode= 'closest',
        #     xaxis= dict(
        #         title= 'Pop',
        #         ticklen= 5,
        #         zeroline= False,
        #         gridwidth= 2,
        #     ),
            yaxis=dict(
                title= 'Feature Importance',
                ticklen= 5,
                gridwidth= 2
            ),
            showlegend= False
        )
        fig = go.Figure(data=data, layout=layout)
        py.iplot(fig, filename='bar-direct-labels')

    #####
    ##### ML SECTION
    #####
            
    def set_xtraintest(self):
        self.X_train = self.train_data[self.features].values
        self.X_test = self.test_data[self.features].values
        
    def get_oof(self, clf, x_train, y_train, x_test):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.NFOLDS, self.ntest))
        # sklearn >= 0.20
        for i, (train_index, test_index) in enumerate(self.kf.split(self.X_train)):
        # sklearn < 0.20
        # for i, (train_index, test_index) in enumerate(self.kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    # Random Forest parameters
    def set_rf_params(self, myParams=None):
        if(myParams is None):
            self.rf_params = {
                    'n_jobs': -1,
                    'n_estimators': 500,
                    'warm_start': True, 
                    #'max_features': 0.2,
                    'max_depth': 6,
                    'min_samples_leaf': 2,
                    'max_features' : 'sqrt',
                    'verbose': 0
            }
        else:
            self.rf_params = myParams
        
    # Extra Trees Parameters 
    def set_et_params(self, myParams=None):
        if(myParams is None):
            self.et_params = {
                'n_jobs': -1,
                'n_estimators':500,
                #'max_features': 0.5,
                'max_depth': 8,
                'min_samples_leaf': 2,
                'verbose': 0
            }
        else:
            self.et_params = myParams
            
    # AdaBoost parameters
    def set_ada_params(self, myParams=None):
        if(myParams is None):
            self.ada_params = {
                'n_estimators': 500,
                'learning_rate' : 0.75
            }
        else:
            self.ada_params = myParams
            
    # Gradient Boosting parameters
    def set_gb_params(self, myParams=None):
        if(myParams is None):
            self.gb_params = {
                'n_estimators': 500,
                 #'max_features': 0.2,
                'max_depth': 5,
                'min_samples_leaf': 2,
                'verbose': 0
            }
        else:
            self.gb_params = myParams
            
    # Support Vector Classifier parameters 
    def set_svc_params(self, myParams=None):
        if(myParams is None):
            self.svc_params = {
                'kernel' : 'linear',
                'C' : 0.025
            }
        else:
            self.svc_params = myParams
            
    def set_models(self):
        # Create 5 objects that represent our 4 models
        self.rf = SklearnHelper(clf=RandomForestClassifier, seed=self.SEED, params=self.rf_params)
        self.et = SklearnHelper(clf=ExtraTreesClassifier, seed=self.SEED, params=self.et_params)
        self.ada = SklearnHelper(clf=AdaBoostClassifier, seed=self.SEED, params=self.ada_params)
        self.gb = SklearnHelper(clf=GradientBoostingClassifier, seed=self.SEED, params=self.gb_params)
        self.svc = SklearnHelper(clf=SVC, seed=self.SEED, params=self.svc_params)
        
        
    def oof_traintest(self):
        # Create our OOF train and test predictions. These base results will be used as new features (self.et, x_train, y_train, x_test)
        self.et_oof_train, self.et_oof_test = self.get_oof(self.et, self.X_train, self.y, self.X_test)         # Extra Trees
        self.rf_oof_train, self.rf_oof_test = self.get_oof(self.rf, self.X_train, self.y, self.X_test)         # Random Forest
        self.ada_oof_train, self.ada_oof_test = self.get_oof(self.ada, self.X_train, self.y, self.X_test)      # AdaBoost 
        self.gb_oof_train, self.gb_oof_test = self.get_oof(self.gb, self.X_train, self.y, self.X_test)         # Gradient Boost
        self.svc_oof_train, self.svc_oof_test = self.get_oof(self.svc, self.X_train, self.y, self.X_test)      # Support Vector Classifier
        print("Training is complete")
        
    def set_features(self, myFeatures=None):
        if(myFeatures == None):
            self.features = [
                "female",
                "male",
                "Age",
                "male_adult",
                "female_adult",
                "child",
                "TitleCat",
                "Pclass",
                #"Ticket_Id",
                "NameLength",
                "CabinType",
                "CabinCat",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
                "Surname_Members",
                "Ticket_Members",
                "Fam_Size"
                #"Ticket_perishing_women",
                #"Ticket_surviving_men",
                #"Surname_perishing_women",
                #"Surname_surviving_men",
            ]
        else:
            self.features = myFeatures
            
    def new_feature_importance(self):
        # (x_train,y_train)
        self.rf_feature  = self.rf.feature_importances(self.X_train,self.y)
        self.et_feature  = self.et.feature_importances(self.X_train, self.y)
        self.ada_feature = self.ada.feature_importances(self.X_train, self.y)
        self.gb_feature  = self.gb.feature_importances(self.X_train, self.y)
        
        cols = self.train_data[self.features].columns.values
        # Create a dataframe with features
        self.feature_dataframe = pd.DataFrame( {'features': cols,
            'Random Forest feature importances': self.rf_feature,
            'Extra Trees  feature importances': self.et_feature,
            'AdaBoost feature importances': self.ada_feature,
            'Gradient Boost feature importances': self.gb_feature
        })
        self.feature_dataframe_mean = self.feature_dataframe.copy()
        self.feature_dataframe_mean['mean'] = self.feature_dataframe_mean.mean(axis= 1) # axis = 1 computes the mean row-wise
        
        self.feature_dataframe_mean.head(5)
        #print(self.feature_dataframe_mean)
        
    def base_predictions_train(self):
        self.base_predictions_train = pd.DataFrame( 
            {'RandomForest': self.rf_oof_train.ravel(),
             'ExtraTrees': self.et_oof_train.ravel(),
             'AdaBoost': self.ada_oof_train.ravel(),
             'GradientBoost': self.gb_oof_train.ravel(),
             'SVC': self.svc_oof_train.ravel(),
            })
        
        self.base_predictions_train.head()    
    
    def concatenate_base_predictions(self):
        self.X_train = np.concatenate(( self.et_oof_train, self.rf_oof_train, self.ada_oof_train, self.gb_oof_train, self.svc_oof_train), axis=1)
        self.X_test = np.concatenate(( self.et_oof_test, self.rf_oof_test, self.ada_oof_test, self.gb_oof_test, self.svc_oof_test), axis=1)
        #self.X_train = np.concatenate(( self.et_oof_train, self.rf_oof_train, self.ada_oof_train, self.gb_oof_train), axis=1)
        #self.x_test = np.concatenate(( self.et_oof_test, self.rf_oof_test, self.ada_oof_test, self.gb_oof_test), axis=1)

    def test_XGBoost(self, param_test=None):       
        gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                                         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                                                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

        gsearch1.fit(self.X_train, self.y)
        print(gsearch1.cv_results_)
        print(gsearch1.best_params_) 
        print(gsearch1.best_score_)

    def run_XGBoost(self, param_run=None):
        if(param_run is None):
            param_run = {
                learning_rate : 0.1,
                n_estimators : 5000,
                max_depth : 4,
                min_child_weight : 12,
                gamma : 0.0,                        
                subsample : 0.6,
                colsample_bytree : 0.6,
                objective : 'binary:logistic',
                nthread : -1,
                reg_alpha : 1,
                scale_pos_weight : 1    
            }
            
        gbm = xgb.XGBClassifier(*+param_run).fit(self.X_train, self.y)
        evals_result = gbm.evals_result()
        print(evals_result)
        
        self.predictions = gbm.predict(self.X_test)
        
    def submit(self):
        PassengerId = np.array(self.test_data["PassengerId"]).astype(int)
        my_prediction = pd.DataFrame(self.predictions, PassengerId, columns=["Survived"])
        from datetime import datetime
        filename = "my_submission"+str(datetime.now())[11:19].replace(":","_")+".csv"
        
        my_prediction.to_csv(filename, index_label=["PassengerId"])
        
        #output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
        #output.to_csv('my_submission.csv', index=False)
        print("Your submission was successfully saved!") 
        
    def old_feature_importance(self, myFeatures=None):
        if(myFeatures != None):
            self.features = myFeatures
            
        for datasets, names in zip(self.combine, self.wichname):   
            if(names=="Training data"):
                selector = SelectKBest(f_classif, k=len(self.features))
                selector.fit(datasets[self.features], self.target)
                scores = -np.log10(selector.pvalues_)
                indices = np.argsort(scores)[::-1]
                print("Features importance :")
                for f in range(len(scores)):
                    print("%0.2f %s" % (scores[indices[f]],self.features[indices[f]]))
         
    def setModel(self):
        self.rfc = RandomForestClassifier(
            n_estimators=3000, min_samples_split=4, class_weight={0: 0.745, 1: 0.255}
        )
        
    def cross_val_score(self, myFeatures=None):
        if(myFeatures != None):
            self.features = myFeatures
            
        kf = KFold(n_splits=3, random_state=1)
        scores = cross_val_score(self.rfc, self.train_data[self.features], self.target, cv=kf)
        print(
            "Accuracy: %0.3f (+/- %0.2f) [%s]"
            % (scores.mean() * 100, scores.std() * 100, "RFC Cross Validation")
        )
        
    def full_score(self, myFeatures=None):
        if(myFeatures != None):
            self.features = myFeatures
        
        self.rfc.fit(self.train_data[self.features], self.target)
        score = self.rfc.score(self.train_data[self.features], self.target)
        print("Accuracy: %0.3f            [%s]" % (score * 100, "RFC full test"))
        
        importances = self.rfc.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(len(self.features)):
            print(
                "%d. feature %d (%f) %s"
                % (
                    f + 1,
                    indices[f] + 1,
                    importances[indices[f]] * 100,
                    self.features[indices[f]],
                )
            )
            
    def run(self, myFeatures=None):
        if(myFeatures != None):
            self.features = myFeatures
         
        # Moved to a metodh set_xtraintest()
        #X_train = self.train_data[self.features]
        #X_test = self.test_data[self.features]
        
        
        predictions = self.rfc.predict(self.X_test)
        #predictions
        PassengerId = np.array(self.test_data["PassengerId"]).astype(int)
        my_prediction = pd.DataFrame(predictions, PassengerId, columns=["Survived"])

        my_prediction.to_csv("my_submission.csv", index_label=["PassengerId"])
        
        #output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
        #output.to_csv('my_submission.csv', index=False)
        print("Your submission was successfully saved!")