[//]: # (Image References)
[image1]: ./figures/simple-model-result.png "Simple Model Training Output 
- Oscillating loss & low accuracy"

5. I took the advices from my mentor and start working with this project once
I finished the Tensorflow lab. 

The first model I used was a simple, one-layer neural network, just as the one
I practised in the Tensorflow lab. Although this model did a nice job in the 
MNIST dataset, it performed badly in the traffic sign classifier project.
 
I could hardly get a meaningful result from training. Below was the loss & 
 accuracy graph after 10 epochs of training:

![Train output for simple NN][image1]

The loss was oscillating, and the accuracy was pretty low.

I though it was a sign the problem was not quite a linearly separable one, so
I turned for the deep neural network.
