## Assignment 2

The second assignment focuses on creation of a decision tree based algorithm which can function as a __Wordle_Solver__. The tree based algorithm which uses the information theoretic concept of __Shannon Entropy__ and __Information Gain__ to calculate the best possible guess made at any step while playing the wordle inorder to be able to predict the correct word in the mimum possible number of queries provided prediction of the _correct characters_ at the _correct locations_ reveals hints which helps the algorithm to predict words faster. 

The __Judging Criteria__ includes:
1. `Average umber of queries` needed to predict words in the dictionary/vocabulary.
2. `Size` of the decision tree model in bytes.
3. `Training Time`
4. `Expected Win Rate` on the sample test set.

The stats for our model are listed: 

-  `Average umber of queries` = 3.99 ~ 4 queries
-  `Size of Decision Tree` = 698574.0B
-  `Training Time` = 11.2780 secs
-  `Expected Win Rate` = 100%