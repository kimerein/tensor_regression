# tensor_regression
Pytorch functions for performing tensor regression. Still under development. Message me if you'd like to help in its development: RichHakim@gmail.com 

These methods are very useful for regression of time series data. I use them to classify and predict neural and behavioral time series data.


## Standard CP tensor regression:

  - where `y = inner_product(X, B) + bias` 
  - where `X` is N-dimensional, `B` has shape `( X.shape[1:] )`, `y` is of length `X.shape[0]`, and `bias` is a scalar 
  - where `B = outer_product(B_cp)` 
  - where `B_cp` is a Kruskal tensor represented as a list of 2-D arrays, each having shapes 
    `[ (i , rank) for i in X.shape[1:] ]`
  - where rank is a low integer
  
## Multinomial CP tensor regression:

  - where `y = softmax(inner_product(X, B)) + bias` 
  - where `X` is N-dimensional, `B` has shape `([ X.shape[1:] ] + [n_classes])`, `y` is of shape `(X.shape[0], n_classes)`, and `bias` is length `n_classes`
  - where `B = outer_product(B_cp)` 
  - where `B_cp` is a Kruskal tensor represented as a list of 2-D arrays, each having shapes 
    `[ (i , rank) for i in X.shape[1:] ] + [(n_classes , rank)]` 
  - where rank is a low integer
