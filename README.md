# UCI-dry-beans

Spawn neural network object:
<name> = NeuralNetwork()
^^ what can be tweaked:
- learning_rate
- mb_size (size of mini batches, set at default 32)
- amount of iterations
  
Train neural network on training set and testing set:
say we named our neural network "nn":
nn.fit(Xtrain, ytrain) -- for training set
nn.fit(Xtest, ytest) -- for testing set
  
Graph plot of loss curve after training:
nn.plot_loss()

Predict w/ neural network (I think data needs to be standardized with StandardScaler):
nn.predict_nr(sample, could be Xtrain or Xtest) -- predict without rounding
nn.predict_r(sample) -- predict with rounding
  
Check accuracy w/ neural network:
For training set:
nn.acc(nn.out_vec(ytrain), nn.predict_r(Xtrain))
For testing set:
nn.acc(nn.out_vec(ytest), nn.predict_r(Xtest))
