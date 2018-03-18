import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path,low_memory= False, parse_dates = ['saledate'])
    X = df
    y = df.SalePrice
    return X,y

def year_made_hist(YearMade,title,ax):
    #fig,ax = plt.subplots(1,1)
    t = ax.hist(YearMade,alpha = .7,normed = 1);
    ax.grid(alpha = .5, color = 'r', linestyle = ':')
    ax.set_xlabel('Year Made')
    ax.set_ylabel('Count')
    ax.set_title(title);
    return t

def scatter_year_made(x,y,title,ax):
    t = ax.scatter(x,y,alpha = .051);
    ax.grid(alpha = .5, color = 'r', linestyle = ':')
    ax.set_xlabel('Year Made')
    ax.set_ylabel('Price')
    ax.set_title(title)
    return t
