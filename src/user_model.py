'''
Reference implementation of node2vec. 
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import os
from nodevectors import Node2Vec


def main():
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	
	g2v = Node2Vec.load('node2vec.zip')

	# Save model to gensim.KeyedVector format

	#g2v.save_vectors("wheel_model.bin")
	
	a = g2v.model
	
	

	# load in gensim
	#from gensim.models import KeyedVectors
	#model = KeyedVectors.load_word2vec_format("wheel_model.bin")
	
	#model[str(43)] # need to make nodeID a str for gensim
	
	game_name = "PLAYERUNKNOWN'S BATTLEGROUNDS"
	
	similar_list = a.wv.most_similar(positive=[game_name], topn=10)
	print(game_name)
	print(similar_list)

	
	
	
if __name__ == "__main__":
	main()
	#read_graph()