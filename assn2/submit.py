import numpy as np

def my_fit( words, verbose = False ):
	dt = Tree( min_leaf_size = 1, max_depth = 15 )
	all_words = words
	dt.fit(all_words, verbose )
	return dt



class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, all_words, verbose = False ):
		self.root = Node( depth = 0, parent = None )
		if verbose:
			print( "root" )
			print( "└───", end = '' )
		# The root is trained with all the words
		self.root.fit( all_words, my_words_idx = np.arange( len( all_words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, verbose = verbose )


class Node:
	# A node stores its own depth (root = depth 0), a link to its parent
	# A link to all the words as well as the words that reached that node
	# A dictionary is used to store the children of a non-leaf node.
	# Each child is paired with the response that selects that child.
	# A node also stores the query-response history that led to that node
	# Note: my_words_idx only stores indices and not the words themselves
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		# self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
		self.is_root = True
	
	# Each node must implement a get_query method that generates the
	# query that gets asked when we reach that node. Note that leaf nodes
	# also generate a query which is usually the final answer
	def get_query( self ):
		return self.query_idx
	
	# Each non-leaf node must implement a get_child method that takes a
	# response and selects one of the children based on that response
	def get_child( self, response ):
		# This case should not arise if things are working properly
		# Cannot return a child if I am a leaf so return myself as a default action
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			# This should ideally not happen. The node should ensure that all possibilities
			# are covered, e.g. by having a catch-all response. Fix the model if this happens
			# For now, hack things by modifying the response to one that exists in the dictionary
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child 
	
	# Dummy leaf action -- just return the first word
	def process_leaf( self, my_words_idx):
		return my_words_idx[0]
	
	def reveal( self, all_word, query ):
		# Find out the intersections between the query and the word
		mask = [ *( '_' * len( all_word ) ) ]
		
		for i in range( min( len( all_word ), len( query ) ) ):
			if all_word[i] == query[i]:
				mask[i] = all_word[i]
		
		return ' '.join( mask )
	
	def calc_entropy(self, my_words_idx, is_array):
		if(is_array):
			n = len(my_words_idx)
			p = np.full(n, 1/n)
			entropy = -np.sum(p * np.log2(p))
		else:
			p = np.full(my_words_idx, 1/my_words_idx)
			entropy = -np.sum(p * np.log2(p))
		return entropy

	def calc_entropy1(self, num):
		p = np.full(num, 1/num)
		entropy = -np.sum(p * np.log2(p))
		return entropy

	def get_entropy(self, all_words, my_words_idx):
		init_entropy = self.calc_entropy(my_words_idx, True)
		max_id = 0
		info_gain_arr = []
		for word_idx in my_words_idx:
			word = all_words[word_idx]
			split_dict = {}
			for idx in my_words_idx:
				mask = self.reveal(all_words[idx], word)
				if mask not in split_dict:
					split_dict[ mask ] = 0
				split_dict[mask]=split_dict[mask]+1
			entropy = -np.sum((value/len(my_words_idx))*self.calc_entropy(value, False) for value in split_dict.values())
			info_gain = init_entropy+entropy
			info_gain_arr.append(info_gain)
		max_id = info_gain_arr.index(max(info_gain_arr))
		max_id = my_words_idx[max_id]
		return max_id


	# Dummy node splitting action -- use a random word as query
	# Note that any word in the dictionary can be the query
	def process_node( self, all_words, my_words_idx,  verbose ):
		# For the root we do not ask any query -- Melbot simply gives us the length of the secret word
		if self.is_root == True:
			query_idx = -1
			query = ""
		else:
			
			# query_idx = np.random.randint( 0, len( all_words ) )
			# query = all_words[ query_idx ]
			query_idx = self.get_entropy(all_words, my_words_idx)
			query = all_words[query_idx]
		
		split_dict = {}
		
		for idx in my_words_idx:
			mask = self.reveal( all_words[ idx ], query )
			if mask not in split_dict:
				split_dict[ mask ] = []
			
			split_dict[ mask ].append( idx )
		
		if len( split_dict.items() ) < 2 and verbose:
			print( "Warning: did not make any meaningful split with this query!" )
		
		return ( query_idx, split_dict )
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    ", verbose = False ):
		self.my_words_idx = my_words_idx
		
		# If the node is too small or too deep, make it a leaf
		# In general, can also include purity considerations into account
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx)
			if verbose:
				print( '█' )
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( all_words, self.my_words_idx, verbose )
			
			if verbose:
				print( all_words[ self.query_idx ] )
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				if verbose:
					if i == len( split_dict ) - 1:
						print( fmt_str + "└───", end = '' )
						fmt_str += "    "
					else:
						print( fmt_str + "├───", end = '' )
						fmt_str += "│   "
				
				# Create a new child for every split
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				self.children[response].is_root = False

				# Recursively train this child node
				self.children[ response ].fit( all_words, split, min_leaf_size, max_depth, fmt_str, verbose )
