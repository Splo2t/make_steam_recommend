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

node_class = dict()


class User():
	def __init__(self, id, game_list):
		self.id = id
		if(type(game_list) == type(list())):
			self.game_list = game_list
		else :
			self.game_list = [game_list]
	def add_game(self, game_name):
		self.game_list.append(game_name)

class Game():
	def __init__(self, id, recommend, user_list, game_real_name):
		self.id = id
		self.recommend = recommend
		self.game_real_name = game_real_name
		if(type(user_list) == type(list())):
			self.user_list = user_list
		else :
			self.user_list = [user_list]
	def add_user(self, user_name):
		self.user_list.append(user_name)

def edge_score(value1, value2):
	
	common_user_count = len((set(value1.user_list) & set(value2.user_list)))
	if common_user_count/min(len(value1.user_list), len(value2.user_list)) >  0.1:
		return 1
	else :
		return 0
	
	return 0
def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()
user_node = dict()
game_node = dict()
def read_graph():
	global node_class
	'''
	Reads the input network in networkx.
	'''
	
	#node_class = dict()
	edgelist = list()
	#class_num = 1
	#class_name_to_num = dict()
	pd.set_option('display.float_format', '{:.2e}'.format)
	review_data = pd.read_csv('drop_review_steam.csv',dtype=np.str)
	#review_data = pd.read_csv('top_test2.csv',dtype=np.str)
	
	for l in range(review_data.shape[0]):
		user_name = review_data.loc[l][-7]
		game_name = review_data.loc[l][3]
		recommend = review_data.loc[l][8] #top_2는는 8, 원본으로는 9
		game_real_name = review_data.loc[l][2]
		if game_name in game_node.keys():
			game_node[game_name].add_user(user_name)
		else :
			game_node[game_name] = Game(game_name, recommend, user_name, game_real_name)
		
		if user_name in user_node.keys():
			user_node[user_name].add_game(game_name)
		else :
			user_node[user_name] = User(user_name,game_name)

		#l = line.strip().split(",")
	
	check_list = []
	for key1, value1 in game_node.items():
		check_list.append(key1)
		for key2, value2 in game_node.items():
			if edge_score(value1, value2) >= 1:
				edgelist.append((key1,key2))
			
				

	G = nx.DiGraph()
	G.add_edges_from(edgelist)
	for edge in G.edges():
		G[edge[0]][edge[1]]['weight'] = 1
	G = G.to_undirected()
	print(G.number_of_nodes())
	print(G.number_of_edges())
	return G

	#print(len(user_node.keys()))
	#print(review_data.shape)

"""
	with open('test.csv', 'r') as f:
		for line in f:
			
			

			if class_name not in class_name_to_num:
				class_name_to_num[class_name] = class_num
				class_num += 1
			node_class[l[-7]] = class_name #class_name_to_num[class_name]
			#for line in f2:
		#		l = line.strip().split()
		#		edgelist.append((l[1],l[0]))

	G.add_edges_from(edgelist)
	for edge in G.edges():
		G[edge[0]][edge[1]]['weight'] = 1
	#G = G.to_undirected()
	return G
"""

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks] # convert each vertex id to a string
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return model

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	
	nx_G = read_graph()
	
	g2v = Node2Vec(
    n_components=32,
    walklen=10
	)
	# way faster than other node2vec implementations
	# Graph edge weights are handled automatically
	g2v.fit(nx_G)

	# query embeddings for node 42
	#g2v.predict(42)

	# Save and load whole node2vec model
	# Uses a smart pickling method to avoid serialization errors
	# Don't put a file extension after the `.save()` filename, `.zip` is automatically added
	g2v.save('node2vec')
	# You however need to specify the extension when reading it back
	
	g2v = Node2Vec.load('node2vec.zip')

	# Save model to gensim.KeyedVector format

	#g2v.save_vectors("wheel_model.bin")
	

	
	a = g2v.model
	
	

	# load in gensim
	#from gensim.models import KeyedVectors
	#model = KeyedVectors.load_word2vec_format("wheel_model.bin")
	
	#model[str(43)] # need to make nodeID a str for gensim
	
	similar_list = a.wv.most_similar(positive=['The Witcher 3: Wild Hunt'], topn=10)
	print('The Witcher 3: Wild Hunt')
	print(similar_list)

	
	
	"""
	
	nx_G = nx.relabel_nodes(nx_G, { n:str(n) for n in nx_G.nodes()})
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	model1 = learn_embeddings(walks)

	node_classification(model1, G.G) #G.G => G를 nx 그래프 형태로 인스턴스화 해놓은 것
	tsne_visualization(model1)
	"""


def tsne_visualization(model):
  global node_class
  node_ids = model.wv.index_to_key  # list of node IDs
  node_subjects = pd.Series(node_class)
  node_targets = node_subjects.loc[node_ids]

  transform = TSNE  # PCA
  trans = transform(n_components=3)
  node_embeddings_3d = trans.fit_transform(model.wv.vectors)

  alpha = 0.7
  label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
  node_colours = [label_map[target] for target in node_targets]

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  #plt.axes().set(aspect="equal")
  ax.scatter(
      node_embeddings_3d[:, 0],
      node_embeddings_3d[:, 1],
      node_embeddings_3d[:, 2],
      c=node_colours,
      cmap="jet",
      alpha=alpha,
  )
  plt.title("{} visualization of node embeddings".format(transform.__name__))
  plt.show()
  plt.savefig("visualization.png")

def node_classification(model, nx_G):
  K = 7
  kmeans = KMeans(n_clusters=K, random_state=0)
  kmeans.fit(model.wv.vectors)

  for n, label in zip(model.wv.index_to_key, kmeans.labels_):
    nx_G.nodes[n]['label'] = label

  for n in nx_G.nodes(data=True):
    if 'label' not in n[1].keys():
      n[1]['label'] = 7
  plt.figure(figsize=(12, 6), dpi=600)
  nx.draw_networkx(nx_G, pos=nx.layout.spring_layout(nx_G), 
  				node_color=[[n[1]['label'] for n in nx_G.nodes(data=True)]], 
					cmap=plt.cm.rainbow,
          node_shape='.',
          font_size='2'
					)
 
  plt.axis('off')
  plt.savefig('img.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
	args = parse_args()
	main(args)
	#read_graph()