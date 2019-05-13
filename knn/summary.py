'''
    Summarize and visualize the data to develop  data intuition

'''
import pandas as pd
import seaborn as sns


# read in the data and 

data_file = 'data.csv'
viz_data = pd.read_csv(data_file)


def plotLatLong(viz_data):
    # plot the lat long of the data
    sns.relplot(x = "latitude", y = "longitude", 
        size = "close_price", sizes = (40,400), palette="muted",
        alpha = .5, data = viz_data)


def plotDist(viz_data, var):
    # plot the lat long of the data
    sns.distplot(viz_data[var])





