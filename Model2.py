from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from collections import defaultdict, Counter
from datetime import timedelta
from numpy.random import randint

class Get_age(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        #print('get age\n','-'*50)
        return self
    def transform(self, X):
        #print('-'*50,'\n','Transforming Age','\n','-'*50)
        Sale_date = pd.to_datetime(X.saledate)#
        X['Calculated_Age'] = Sale_date.dt.year- X['YearMade']
        X.loc[X['YearMade_Fixed']== 1,'Calculated_Age'] = 5
        return X

class Replace_outliers(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        #print('Replace Outliers\n','-'*50)
        self.value = X[X['YearMade'] > 1000].YearMade.mode()
        return self

    def transform(self,X):
        #print('-'*50,'\n','Transforming replace Outliers','\n','-'*50)
        condition = X.YearMade > 1900
        X['YearMade_Fixed'] = 0
        X.loc[~condition, 'YearMade'] = self.value[0]
        X.loc[~condition, 'YearMade_Fixed'] = 1
        return X

class mean_price(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        # print('Mean Price\n','-'*50)
        # Model = X['ModelID']
        # d = {}
        # for model,num in Counter(Model).items():
        #     Mod_df = (X[X['ModelID'] == model].SalePrice)
        #     d[model] = np.mean(Mod_df)

        Model = X['ModelID']
        d = {}
        for model,num in Counter(Model).items():
            d2 = defaultdict(list)
            Mod_df = (X[X['ModelID'] == model].reset_index())
            for i in range(Mod_df.reset_index().shape[0]):

                year = pd.to_datetime(Mod_df.loc[i,'saledate'])
                d2[year].append(Mod_df.loc[i,'SalePrice'])
            d[model] = d2
        self.d = d
        self.p = np.mean(X.SalePrice)
        return self
    def transform(self,X):
        # print('-'*50,'\n','Transforming Mean Price','\n','-'*50)
        # X['MeanPrice'] = 0
        # for model in self.d:
        #     idx = X['ModelID'] == model
        #     X.loc[idx,'MeanPrice'] = self.d[model]
        # idx2 = X['MeanPrice'] == 0
        # X.loc[idx2,'MeanPrice'] =self.p


        X['MeanPrice'] = 0
        X = X.reset_index()
        for i in range(X.shape[0]):
            sd = pd.to_datetime(X.loc[i,'saledate'])
            Means = []
            timed = []
            if X.loc[i,'ModelID'] in self.d:
                for item in self.d[X.loc[i,'ModelID']].items():
                    if (item[0] < sd):
                        td = (sd - item[0]).total_seconds()
                        Means.append(np.mean(item[1]))
                        #timed.append(int(td))
                #timed = list(reversed(timed))
                #timed = np.array(timed)/31557600
                #weights = np.exp(timed)/np.sum(np.exp(timed))
                if len(Means) > 0:
                    x = np.linspace(1,len(Means),len(Means))
                    Mean = np.average(np.array(Means), axis=None, weights=list(reversed(x)))
                    X.loc[i,'MeanPrice']= Mean
                else:
                #Mean = np.mean(np.array(Means)*list(reversed(x)))
                    X.loc[i,'MeanPrice']= self.p
            else:
                X.loc[i,'MeanPrice'] = self.p
            if i%10000 == 0:
                print(i)

        cond = np.isnan(X.MeanPrice)
        X.loc[cond,'MeanPrice']  = self.p
        # X['MeanPrice2'] = 0
        return X

class Get_dummies(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        ProductSize = X.ProductSize
        ProductSize[ProductSize == 'Large'] = 6
        ProductSize[ProductSize == 'Large / Medium'] = 5
        ProductSize[ProductSize == 'Medium'] = 4
        ProductSize[ProductSize == 'Small'] = 3
        ProductSize[ProductSize == 'Compact'] = 2
        ProductSize[ProductSize == 'Mini'] = 1
        ProductSize[ProductSize == 'None or Unspecified'] = np.nan
        m = ProductSize[~np.isnan(np.array(ProductSize, dtype=np.float64))].mean()
        self.m = m
        return self


        # S = pd.get_dummies(X['state'])
        # X = X.join(S)
        # self.colum = np.array(X.columns)
        # return self
    def transform(self,X):
        #print('-'*50,'\n','Transforming dummies','\n','-'*50)
        ProductSize = X.ProductSize
        ProductSize[ProductSize == 'Large'] = 6
        ProductSize[ProductSize == 'Large / Medium'] = 5
        ProductSize[ProductSize == 'Medium'] = 4
        ProductSize[ProductSize == 'Small'] = 3
        ProductSize[ProductSize == 'Compact'] = 2
        ProductSize[ProductSize == 'Mini'] = 1
        ProductSize[ProductSize == 'None or Unspecified'] = np.nan
        ProductSize[np.isnan(np.array(ProductSize, dtype=np.float64))]= self.m
        X['ProductSize'] = ProductSize
        return X

        # S = pd.get_dummies(X['state'])
        # X = X.join(S)
        # for i in range(5,X.shape[1]):
        #     if list(X.columns[i])  != list(self.colum[i]):
        #         X.insert(i, self.colum[i], 0)
        # return X

        #X = X.set_index('SalesID')[self.columns].sort_index()


class Only_cols(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        #print('Only Cols\n','-'*50)
        return self
    def transform(self,X):
        #print('-'*50,'\n','Transforming Columns','\n','-'*50)
        c = ['MeanPrice','Calculated_Age','ProductSize']
       #  c = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
       # 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
       # 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
       # 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
       # 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
       # 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
       # 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
       # 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina',
       # 'South Dakota', 'Tennessee', 'Texas', 'Unspecified', 'Utah', 'Vermont',
       # 'Virginia', 'Washington', 'West Virginia', 'Wisconsin',
       # 'Wyoming','MeanPrice','Calculated_Age','YearMade','YearMade_Fixed']

        X = X[c]

        #X = X.set_index('SalesID')[self.columns].sort_index()
        return X



def rmsle(y_hat, y):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    """

    log_diff = np.log(y_hat) - np.log(y)
    return np.sqrt(np.mean(log_diff**2))

if __name__ == '__main__':
    print('-'*50,'\n','Running Linear Regression.  Lets go baby','\n','-'*50,)
    print('-'*50,'\n','Loading Data','\n','-'*50,)
    df = pd.read_csv('data/Train.csv',low_memory = False,parse_dates = ['saledate'])
    df = df.set_index('SalesID').sort_index()
    y = df.SalePrice


    p = Pipeline([
        ('Replace_outliers', Replace_outliers()),
        ('mean_price',mean_price()),
        ('Get_age',Get_age()),
        ('Get_dummies',Get_dummies()),
        ('Only_cols', Only_cols()),
        ('lm', Lasso(alpha=2154.4346900318865))
        ])
        #('Get_dummies',Get_dummies()),
    params = {'lm__alpha':.5}
    clf = p.fit(df, y)

    print('Loading in Test Data','\n','-'*50,)
    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')


    Salesid = test.SalesID
    print('Predicting off of Test Data')
    test_predictions = clf.predict(test)
    test_predictions = test_predictions

    print('prediction finished results coming!!')
    test['SalePrice'] = test_predictions
    test.set_index('SalesID')
    outfile = 'data/Elliott_version.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)

    test_solution = pd.read_csv('data/do_not_open/test_soln.csv')
    test_solution.set_index('SalesID')
    results = (rmsle(test_solution.SalePrice,test_predictions))
    print(results)

    #
    # print('\n\n\nHello\n\n\n','-'*50,'\n\n\n\nHello\n\n\n')
