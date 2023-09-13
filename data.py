from keras.datasets import mnist
import numpy as np
import pickle


# ------ Feature engineering

def one_hot_encoding(labels):
	labels_one_hot = np.zeros((len(labels), 10))
	for i in range(len(labels)):
		labels_one_hot[i][labels[i]] = 1
	return labels_one_hot

def data_engineer(data_X):
	for i in range(len(data_X)):
		data_X[i][data_X[i] < 64] = 0
		data_X[i][data_X[i] >= 64] = 1



(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X.resize((train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
test_X.resize((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))

data_engineer(train_X)
data_engineer(test_X)

train_y = one_hot_encoding(train_y)

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

data = (train_X, train_y), (test_X, test_y)

with open('data', 'wb') as flow:
	pickle.dump(data, flow)
