import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def swap(X, Y):
    X[0], Y[0]  = Y[0], X[0]
    X[1], Y[1]  = Y[1], X[1]
    X[2], Y[2]  = Y[2], X[2]
    X[3], Y[3]  = Y[3], X[3]

def compare(a,b):
    n1 = 1*a[3] + 2*a[2] + 4*a[1] + 8*a[0]
    n2 = 1*b[3] + 2*b[2] + 4*b[1] + 8*b[0]

    res = 1 if n1 > n2 else 0
    return (n1, n2, res)

def preprocess(X):
    for row in X:
        res = compare(row[64:68], row[68:72])
        if(int(res[2]) == 1):
            swap(row[64:68], row[68:72])
            row[72] = 1 - row[72]
    return X

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

	Z_train = preprocess(Z_train)

	groups = defaultdict(list)
	outputs = defaultdict(list)
	model = defaultdict(list)

	score,flag = 0,0

	for e in Z_train:
		i, j, flag = compare(e[64:68], e[68:72])
		groups["{}->{}".format(i,j)].append(e[0:64])
		outputs["{}->{}".format(i,j)].append(e[72])

	for key in groups.keys():
		i = np.array(groups[key])
		o = np.array(outputs[key])
		# clf = LinearSVC(loss="squared_hinge", dual = True, C = 0.7, max_iter = 700)
		clf = LogisticRegression(penalty="l1", warm_start = True, verbose = True, n_jobs = -1, max_iter = 1500)
		clf.fit(i,o)
		model[key] = clf
		score += model[key].score(i,o)/120
		flag += 1

	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
	dim = X_tst.shape

	score, flag = 0, 0
	pred = np.zeros(dim[0])

	for a in range(dim[0]):
		e = X_tst[a]
		i, j, flag = compare(e[64:68], e[68:72])
		if(flag == 1):
			m = model["{}->{}".format(j,i)].predict([e[0:64]])
			pred[a] = 1-m
		else:
			m = model["{}->{}".format(i,j)].predict([e[0:64]])
			pred[a] = m
	return pred
