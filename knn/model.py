
'''
    Build out the KNN model to predict prices
    for observation
'''
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import heapq
import time
import pdb


def cleanData(data):
    '''
        Clean data before train and infer step
    '''
    clean_data = filterNegPrices(data)
    clean_data = createLogPrice(clean_data)
    clean_data = createDate(clean_data)
    return clean_data

def filterNegPrices(data):
    '''
        Input: data in pandas format
        Output: filter out negative price or zero data
    '''
    is_nonzero_price = data['close_price']>0
    filter_data = data[is_nonzero_price]
    return filter_data

def createDate(data):
    '''
        Create date filter step
    '''
    data['close_date'] = data['close_date'].apply(
        lambda x: datetime.strptime(
            str(x.split('.')[0]), '%Y-%m-%d %H:%M:%S'))
    data['date'] = data['close_date'].apply(lambda x: datetime.date(x))
    return data

def createLogPrice(data):
    data['log_price'] = np.log(data['close_price'])
    return data


class KNNModel:

    def __init__(self, data, weight = 'mean'):
        self.data = data
        # methods include (mean or inv_dist)
        self.weight = weight


    def runKNN(self, k, price_var):
        self.predictions = [None]
        self.error_rates = [0]
        self.data = self.data.sort_values(['close_date'], ascending = True)
        obs_num = self.data.shape[0]
        t0 = time.time()
        for i in range(1,obs_num):
            prediction, error = self.predictSingleObs(i, k, price_var)
            t1 = time.time()
            if i%10000==0:
                print(t1-t0)
                print(i)
            self.predictions.append(prediction)
            self.error_rates.append(error)
        self.maer = np.median(self.error_rates)
        self.data['prediction'] = self.predictions
        write_filename = 'prediction_' + str(k) + '.csv'
        self.data.to_csv(write_filename)

    # For each observation, find the euclidean distance of lat/long
    def predictSingleObs(self, i, k, price_var):
        # filter out data that occurred prior
        target_data = self.data.iloc[i]
        train_data =  self.data.iloc[:i]
        # filter out training data occuring 6 months prior to target
        # thresh_date = target_data.close_date - timedelta(days = 12*30)
        # train_data = train_data[train_data.close_date > thresh_date]
        # if data is nonzero
        if len(train_data)==0:
            return None, 0
        else:
            train_data = self.createDistanceMeasure(target_data, train_data)
            # [TODO] try out logging prediction price to see if help
            if self.weight == 'mean':
                pred_price = self.findClosestMeanPrice(train_data, k, price_var)
            else:
                pred_price = self.findClosestWeightedPrice(train_data, k, price_var)
            if price_var == 'log_price':
                pred_price = np.exp(pred_price)
            error = self.measureRAE(target_data, pred_price)
            return pred_price, error


    def createDistanceMeasure(self, target_data, train_data):
        '''
            Input: pandas dataset
            Output: training dataset with distance field
        '''
        # square latitude distance
        lat_sqdiff = np.square(
            train_data.loc[:, 'latitude'] - target_data.loc['latitude'])
        long_sqdiff = np.square(
            train_data.loc[:, 'longitude'] - target_data.loc['longitude'])
        train_data['eucl_dist'] = np.sqrt(lat_sqdiff + long_sqdiff)
        return train_data


    def findClosestMeanPrice(self, train_data, k, price_var):
        '''
            Input: training dataset, neighbor size, and column name of relevant
             price variable
            Output: the mean price of k closest neighbor
        '''
        sorted_data = train_data.sort_values(['eucl_dist'], ascending = True)
        w = min(len(sorted_data), k)
        weights = [1/w]*w
        price = np.array(sorted_data[:k][price_var])
        return np.sum(price*weights)

    def findClosestWeightedPrice(self, train_data, k, price_var):
        '''
            Input: training dataset, neighbor size, and column name of relevant
             price variable
            Output: the weighted mean price of k closest neighbor
        '''
        sorted_data = train_data.sort_values(['eucl_dist'], ascending = True)
        dist = sorted_data[:k].eucl_dist
        # generate weights for each obs proportional to inverse of distance
        # then scale weights to that sum equals to 1
        weights = 1/dist
        weights = weights / np.sum(weights)
        price = np.array(sorted_data[:k][price_var])
        return np.sum(price*weights)


    def measureRAE(self, target_data, pred_price):
        '''
            Input: target prediction data and prediction price
            Output: The mean absolute error rate actual closing and predicted
        '''
        actual_price  = float(target_data['close_price'])
        return  np.abs(pred_price - actual_price)/actual_price



def iterMultipleK(data, method):
    '''
        test out different k parameters to find one with lowest median error
    '''
    print('clean data')
    clean_data = cleanData(data)
    knn = KNNModel(clean_data, method)
    price_var = 'close_price'
    # for k from 1 to 10
    k_vals = range(4,5)
    err = []
    for k in k_vals:
        print('running k = %f' %(k))
        knn.runKNN(k, price_var)
        err.append(knn.maer)
    print(k_vals)
    print(err)


if __name__ == '__main__':
    # read data
    print('read data')
    data = pd.read_csv('data.csv')
    method = 'mean'
    iterMultipleK(data, method)
    # avg error with mean weights [0.20681669292178623]

