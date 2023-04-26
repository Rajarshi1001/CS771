import numpy as np
import pickle as pkl

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df ):
	with open( "my_model", "rb" ) as file:
		model = pkl.load( file )
	
	pred = model.predict( df.drop( [ "Time" ], axis = "columns" ).to_numpy() )
	
	return ( pred[ :, 0 ], pred[ :, 1 ] )