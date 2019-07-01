# CSC 4800 - AI shenanigans, team CROWS
# Assignment: Reading JSON training data, training Hebb nodes

import sys
import numpy as np
import pandas as pd


"""
given a HebbCell hc:
	hc( ... ) passes inputs
	hc.train( ... ) passes training output
	hc.activate() generates output

How this fits together:

	hc( inputs ).train( target ).activate() 
does traning 

 	hc( inputs ).activate()
does testing
"""

class HebbNode:
	# slightly pretty printing
	def __str__( self ):
		fmt = "weights: " + str( self.weights )
		fmt += "\n" + "bias: " + str( self.bias )
		return fmt	

	# Step 0
	# constructs a zero'd-out HebbNode with n edges
	def __init__( self, n ):
		self.edgeCount = n
		self.weights = np.zeros( n )
		self.bias = np.float_( 0 )

	# sets a collection x values
	def __call__( self, inputs ):
		# Step 2
		self.inputs = inputs

		return self

	# uses existing self.inputss, trains the HebbNode to target t
	def train( self, t ):
		# Step 3
		self.y = t

		# too much self.attribute calls seems like a code smell
		# I think there must be something smarter here.
		# These assignments are for clarity
		n = self.edgeCount
		ws = self.weights
		xs = self.inputs
		y = self.y
		bias = self.bias

		# Step 4
		# The nugget is here
		# ( no self. cruft )
		for i in range( n ):
			ws[ i ] += ( xs[ i ] * y )
		bias += y

		self.weights = ws
		self.bias = bias

		return self

	# emits a value based on current inputs
	def activate( self ):

		n = self.edgeCount
		ws = self.weights
		xs = self.inputs
		bias = self.bias

		emit = bias  
		for i in range( n ):
			emit += ws[ i ] * xs[ i ]

		theta = 0
#		return 1 if emit > theta else 0		
		return True if emit > theta else False
#		return emit

def main():
	# file handling
	if len( sys.argv ) < 2:
		print( "no file path in arguments\n(correct usage '$ python3 gates.py [input].json')")
		return

	path = sys.argv[ 1 ]

	# code research:
	# https://stackoverflow.com/questions/22366282/python-filenotfound
	# http://www.pfinn.net/python-check-if-file-exists.html
	try:
		with open( path ) as file:
			data = file.read()
	except FileNotFoundError:
		print( "File not found: " + path )
		return
	except IOError:
		print( "Unable to open file: " + path )
		return

	# table is a pandas dataframe
	# Pretty heavywight object!
	# http://pandas.pydata.org/pandas-docs/stable/reference/frame.html
	table = pd.read_json( data )

	hn = HebbNode( 2 )

	# performant but not very dynamic way of exploring a pandas dataframe
	# code research for some more dynamic ideas (like indexing by column label):
	# https://brohrer.github.io/dataframe_indexing.html 

	# Training
	#	HebbNode( inputs ).train( output )
	for row in table.itertuples():
		hn( row[ 1 ]).train( row[ 2 ])

	print( hn )

	# Testing
	# 	HebbNode( inputs ).activate()

	for row in table.itertuples():
		print( hn( row[ 1 ]).activate() )

# This block works, composition of neurons seems like something that should
# be done very carefully, this seems correct but not quite there
# i.e. How can we build net to conjure its inputs by configuration? Should we?

"""
	z1 = HebbNode( 2 )
	z2 = HebbNode( 2 )
	net = HebbNode( 2 )

	# Training
	# (now requires activation of lower layer to feed higher layers)
	#	HebbNode( inputs ).train( output ).activate()

	for row in table.itertuples():
		factors = row[ 1 ]
		target = row[ 2 ]
		v = []
		v.append( z1( factors ).train( target ).activate() )
		v.append( z2( factors ).train( target ).activate() )
		net( v ).train( target )

	print( net )
"""
if __name__ == "__main__":
	main()