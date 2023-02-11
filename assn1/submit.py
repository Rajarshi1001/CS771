import numpy as np
import sklearn
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def compare1(a,b):
    n1 = 1*a[3] + 2*a[2] + 4*a[1] + 8*a[0]
    n2 = 1*b[3] + 2*b[2] + 4*b[1] + 8*b[0]
    return (n1, n2)
################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	groups = defaultdict(list)
	outputs = defaultdict(list)
	model = defaultdict(list)
	labels = []
	data = []
	counter = 0

	for e in Z_train:
		firstSbits = e[64:68]
		lastSbits = e[68:72]
		labels.append(e[72])
		data.append(e[0:64])
		i, j = compare1(firstSbits, lastSbits)
		m1 = min(int(i), int(j))
		m2 = max(int(i), int(j))
		groups["{}->{}".format(m1, m2)].append(e[0:72])
		outputs["{}->{}".format(m1, m2)].append(e[72])
	#     models["{}->{}".format(m1, m2)]=model

	# print(groups["1->12"])
	for key in groups.keys():
		i = np.array(groups[key])
		o = np.array(outputs[key])
		mode = LogisticRegression(max_iter=1000)
		mode.fit(i,o)
	#     print(i.shape)
	#     counter = counter + i.shape[0]
	#     print(o.shape)
		model[key] = mode
		print(model[key].score(i,o))

	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model):
################################
#  Non Editable Region Ending  #
################################
	a = X_tst[64:68]
	b = X_tst[68:72]
	i, j = compare1(a, b)
	m1 = min(int(i), int(j))
	m2 = max(int(i), int(j))
	pred = model["{}->{}".format(m1, m2)].predict(X_tst[:,:72])
	
	# Use this method to make predictions on test challenges
	
	return pred
