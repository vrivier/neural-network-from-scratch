# neural-network-from-scratch
Functionnal multi-layered perceptron hand-coded tested on MNIST dataset

### Usage

This project needs tensorflow to load the dataset. It then only uses numpy (included in tensorflow).  
To test the neural network on the MNIST dataset, you should first run the **data.py** file, which loads the data, preprocesses it and pickles it (~60Mo). In a second time, you can simply run **train-test.py**. This file will load the preprocessed data, instanciate a neural network, and then train it on the data. It also displays the accuracy every 500 examples passed through the network, so you can see it progressively learn. It reaches 80-90% accuracy pretty fast which is satisfying and shows the network works. 

### Data preprocessing

The network only takes flat arrays as input, and MNIST data is 2D (28*28). I had to flatten the data using resize function from numpy.  
I also had a difficulty with the values contained in the MNIST data, that ranges between 0 and 255 for each pixel. Those high values were preventing the network from training, so I chose a threshold under which the pixel would be 0 and abose which the pixel would be 1. This debugged the training. (normalizing by dividing all values by 255 would be a possible improvement)  
Finally, I encoded the labels as one-hot vectors instead of values from 0 to 9 to be in a multiclass classification paradigm. 

### Hyperparameters

The network comes with possibilities to tune its architecture for fine-tuning purpose. Its number of layers, neurons per layer, learning rate, number of epochs can be tuned.

### Ressources

https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/  
https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH  

### Citation
```
@article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  volume={2},
  year={2010}
}
```
