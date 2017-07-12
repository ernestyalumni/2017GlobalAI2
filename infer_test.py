#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs
import sys
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data, wb
import time
import math
import os
from datetime import datetime
from datetime import date
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

test_docs = sys.argv[-1]

writer = pd.ExcelWriter('/home/ubuntu/fakenewsclean.xlsx')

breitbartData = pd.read_csv('/home/ubuntu/doc2vec-master/breitbartclean.csv')
huffingtonData = pd.read_csv('/home/ubuntu/doc2vec-master/huffingtonclean.csv')
voanewsData = pd.read_csv('/home/ubuntu/doc2vec-master/voanewsclean.csv')
rtData = pd.read_csv('/home/ubuntu/doc2vec-master/rtclean.csv')
##########################################################################
#### I can't remember how to create a new pandas dataframe ###############
##########################################################################
##########################################################################
####labeledData = pd.new????##############################################
##########################################################################

#parameters
model="toy_data/model.bin"
#test_docs="toy_data/test_docs.txt"
output_file="toy_data/test_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model) #yup

##########################################################################
####I think this is reading each line of the .txt document to an array?###
##########################################################################
test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]
##########################################################################
####I think this is reading each line of the .txt document to an array?###
##########################################################################




#infer test vectors
output = open(output_file, "w") ### we want to write to a csv or .xlsx instead of .txt ### 
#################################################################
####I think this is processing the .txt document line by line?###
#################################################################
for d in test_docs:
    output.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" ) ## this line should be modified accordingly
#################################################################
####I think this is processing the .txt document line by line?###
#################################################################

output.flush() # once the pandas datapipeline is done, this can be commented out/deleted
output.close() # once the pandas datapipeline is done, this can be commented out/deleted

#Data labeling
#TODO: add new column to the pandas dataframe
#Breitbart = (1,0,0,0)
#Huffington = (0,1,0,0)
#VOANews = (0,0,1,0)
#RT.com = (0,0,0,1)
labeledData.to_excel(writer,'Sheet1')
writer.save()


