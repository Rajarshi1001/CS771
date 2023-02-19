
## __Assignment 1__:

Cracking a __XORRO Ring Oscillator PUF__ using Cascaded Linear Models. Here we have devised a simple Linear Model which is able to model a __XORRO PUF__ which essentially consists of two XORROs which is connected to a counter that determines which XORRO has the highest frequency. 


Here, we have used a collection of 120 Linear models mapping from a specific _key_ which is detetermined from the first `S` bits and the last `S` bits in the training datasets `train.dat`. Each key has been mapped to a linear model namely **LogisticRegression** and **LinearSVC**, performing a comparative analysis between the two linear models with respect to various *hyperparameters* like `tol`, `penalty = [l1, l2]`, `C`, `max_iter`.

The **LogisticRegression** model with a desired set of a hyperparameters is able to achieve an accuracy of `94.98%` on the test dataset.

All the details are mentioned in the Assignment Report.