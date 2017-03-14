[//]: # (Image References)
[image1]: ./figures/simple-model-bad-result.png "Linear Model Training Output 
- Oscillating loss & low accuracy"
[image2]: figures/simple-model-result.png "Linear Model Training Output with
shuffling train set first"

5. I took the advices from my mentor and start working with this project once
I finished the Tensorflow lab. 

The first model I used was a simple, one-layer neural network, just as the one
I practised in the Tensorflow lab. At first, I assumed the train set was already
randomized and used it without shuffling. That gave me a bad model.
 
Below was the loss & accuracy graph after 10 epochs of training:

![Train output for simple NN - Oscillating][image1]

With my mentor's help I figured out my mistake and re-trained my model. 
That was a lot better:

![Trian output for simple NN - Just fine][image2]