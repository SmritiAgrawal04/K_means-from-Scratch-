
# coding: utf-8

# In[1]:


import numpy as np
import glob
import pickle
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import svm
from numpy import array 
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


# In[2]:


class Cluster:
    def split_data(self, dataset):
        length= int(0.2*len(dataset))
        train_data = dataset[length:]
        valid_data = dataset[:length]

        return train_data.to_numpy(), valid_data.to_numpy()
    
    def prepare_data(self, path):
        data= []
        filenames= []
        files= glob.glob(path)
        for file in files:
            file_name= (np.asarray(file.split("/")))[-1]
#             params= (np.asarray(params.split("_")))[-1]
            filenames.append(file_name)
            f=open(file, 'r', encoding='utf-8', errors='ignore')  
            data.append(np.asarray(f.read()))
            f.close() 

        dataset= pd.DataFrame(filenames)
        df= pd.DataFrame(np.asarray(data))
        dataset= pd.concat([dataset, df], axis= 1)
        return dataset.to_numpy()
    
    def cluster(self, test_path):
        data= []
        labels= []
        path= '/home/smriti/CourseWork/SEMESTER-2/SMAI/Assignment_2/Datasets/Question-6/dataset/*.txt'
        dataset= self.prepare_data(path)
        #separate the labels and content from train data
        filenames= dataset[:,0]
        dataset= dataset[:,1]

        X= self.vectorize(dataset)
        X_tfidf= self.transform(X)
        result =self.Kmeans_Clustering(X_tfidf, filenames)
        return result
        
    def vectorize(self, dataset):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(dataset)
        X.shape
        return X
    
    def transform(self, X):
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X)
        X_train_tfidf.shape
        return X_train_tfidf
        
    def check_allocation(self, train_labels, cluster_allocated):
        total_correct= 0
        for cluster in cluster_allocated:
            frequency= []
            for point in cluster:
                frequency.append(train_labels[point])

            max_freq= max(set(frequency), key = frequency.count)
            correct= frequency.count(max_freq)
            total_correct += correct

        return total_correct/len(train_labels)
    
    def get_result(self, cluster_allocated, file_num):
        cluster_num =0
        for cluster in cluster_allocated:
            if(cluster.count(file_num) >0):
                return cluster_num
            cluster_num +=1
        
    def Kmeans_Clustering(self, X_tfidf, filenames):
        centroids= self.choose_centroids(X_tfidf)
        X_tfidf= X_tfidf.toarray()
        
        iterations =0
        while iterations< 10:
            centroids, cluster_allocated= self.KMeans(centroids, X_tfidf)
            iterations +=1
        result= {}
        file_num=0
        for file in filenames:
            result[file]= self.get_result(cluster_allocated, file_num)
            file_num +=1
#         accuracy= self.check_allocation(filenames , cluster_allocated)*100

        return result

        
    def choose_centroids(self, X_tfidf):
        c1= randint(0, X_tfidf.shape[0])
        c2= randint(0, X_tfidf.shape[0])
        c3= randint(0, X_tfidf.shape[0])
        c4= randint(0, X_tfidf.shape[0])
        c5= randint(0, X_tfidf.shape[0])

        c1= X_tfidf[c1, :].toarray()
        c2= X_tfidf[c2, :].toarray()
        c3= X_tfidf[c3, :].toarray()
        c4= X_tfidf[c4, :].toarray()
        c5= X_tfidf[c5, :].toarray()

        centroids= [c1,c2,c3,c4,c5]
        
        return centroids
    
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        row1= row1.flatten()
        for i in range(0,row1.shape[0]):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)
    
    def KMeans(self, centroids, data): 
        flag= 0
        for center in centroids:
            dist= []
            for point in range (0, data.shape[0]):
                dist.append(self.euclidean_distance(np.array(center).T, data[point, :]))
            if flag ==0:
                distances= pd.DataFrame(dist)
                flag= 1
            else:
                df= pd.DataFrame(dist)
                distances= pd.concat([distances, df], axis= 1)


        distances= distances.to_numpy()
        point_mapping= {'1':[], '2':[], '3':[], '4':[], '5':[]}
        cluster_allocated= [[],[],[],[],[]]

        for i in range (0, len(distances)):
            min_dist= np.amin(distances[i, :])
            dist_list= distances[i, :].tolist()
            index= dist_list.index(min_dist)+1
            if len(point_mapping[str(index)])==0:
                point_mapping[str(index)]= pd.DataFrame(data[i,:])
            else:
                df= pd.DataFrame(data[i,:])
                point_mapping[str(index)]= pd.concat([point_mapping[str(index)], df], axis= 1)
            cluster_allocated[index-1].append(i)        

        points_c1= point_mapping['1'].to_numpy().T
        points_c2= point_mapping['2'].to_numpy().T
        points_c3= point_mapping['3'].to_numpy().T
        points_c4= point_mapping['4'].to_numpy().T
        points_c5= point_mapping['5'].to_numpy().T

        c1= []
        c2= []
        c3= []
        c4= []
        c5 =[]
        
        for cordinate in range(0, data.shape[1]):

            c1.append(np.mean(points_c1[:,cordinate]))

            c2.append(np.mean(points_c2[:,cordinate]))

            c3.append(np.mean(points_c3[:,cordinate]))

            c4.append(np.mean(points_c4[:,cordinate]))

            c5.append(np.mean(points_c5[:,cordinate]))


        centroids= [c1,c2,c3,c4,c5]
        return centroids, cluster_allocated


# In[3]:


# from q6 import Cluster as cl
cluster_algo = Cluster()
# You will be given path to a directory which has a list of documents. You need to return a list of cluster labels for those documents
predictions = cluster_algo.cluster('./Datasets/Question-6/dataset/*.txt') 

# print (predictions)
'''SCORE BASED ON THE ACCURACY OF CLUSTER LABELS'''

