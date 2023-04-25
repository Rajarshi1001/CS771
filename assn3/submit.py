import numpy as np
import pickle as pkl

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df ):
    	
	X = np.array(df.iloc[:, 1:])
	# Load your model file

	model = pkl.load(open("knn_model.pkl", "rb"))
	test_preds = np.transpose(model.predict(X))
	
	# Make two sets of predictions, one for O3 and another for NO2
	pred_o3, pred_no2 = test_preds[0], test_preds[1]
	
	# Return both sets of predictions
	return ( pred_o3, pred_no2 )