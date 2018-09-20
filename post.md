TL for TF (Transfer learning for Tensorflow)

“What is it?”

Transfer learning is a technique for efficiently, partially retraining an (often deep) neural network.

“How is it different from non-transfer learning?”

Two pictures help illustrate:

A fully trained neural net takes input value(s) in an initial layer and then sequentially feeds this information forward until, crucially, some second-to-last layer has constructed a high level model. The full training of the model references the optimization of weight and bias terms used in each connection, labelled in green above. This second-to-last layer is referred to as a bottleneck layer. At last, this bottleneck layer pushes values (in a regression model) or one-hot probabilities (in classification) to our final network layer.




In transfer learning, we take all of the weights and biased learned from a prior model training in all layers except the final layer as fixed. We then simply retrain weights and biases for one layer only (the bottleneck layer). In the diagram above, we take the red connections as fixed, and only now retrain the last layer of green connections.






“Why would I use it?”

Transfer learning confers two major benefits:

Speed of training
Efficient learning with less data

By only retraining our final layer, we’re performing a far less computationally expensive optimization (learning hundreds or thousands of parameters, instead of millions).
Open source models like Inception v3 trained 25 million parameters on best in class hardware. As a result, these nets have well-trained parameters and have produced bottleneck layers that have highly optimized representations of the input data. So while you likely could not have trained a high-performing model from scratch with your own limited computing and data resources, you can leverage the work of others to force-multiply your performance.

“OK, how do I use it?

If you prefer, Tensorflow already has this handy command line demo for retaining some of their models.

But let’s look at some Python code to get slightly more into the weeds (but not too far - don’t want to get lost down there!).

First, we need a pretrained model to start. Keras has a bunch of pretrained models, (but they all seem to be image classifiers and nothing else?). We’ll use the InceptionV3 model to start.

```
from keras.applications import InceptionV3
from keras.models        import Model
```  

As noted above, we want to retain the first n-1 layers of the model, and just retrain a final layer. 

```
bottleneck_input   = original_model.get_layer(index=0).input
bottleneck_output = original_model.get_layer(index=-2).output
bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)
```

Here, we get the input from the first layer (index = 0) of the Inception model. If we `print(model.get_layer(index=0).input)`, we see `Tensor("input_1:0", shape=(?, ?, ?, 3), dtype=float32)` -- this indicates that our model is expecting some indeterminate amount of images as input, of an unspecified height and width, with 3 rbg channels. This, too, is what we want as the input for our bottleneck layer.

We see `Tensor("avg_pool/Mean:0", shape=(?, 2048), dtype=float32)` as the output of our bottleneck, which we accessed by referencing the second to last model layer. In this instance, the Inception model has learned a 2048 dimensional representation of any image input, where we can think of these 2048 dimensions as representing crucial components of an image that are essential to classification.

Lastly, we instantiate a new model with the original image input and the bottleneck layer as output: `Model(inputs=bottleneck_input, outputs=bottleneck_output)`.

Next, we need to set each layer in the pretrained model to untrainable -- essentially we are freezing the weights and biases of these layers and keeping the information that was already learned.

```
for layer in bottleneck_model.layers:
	layer.trainable = False
```

Now, we make a new `Sequential()` model, starting with our previous building block and then making a minor addition:

```
new_model = Sequential()
new_model.add(bottleneck_model)
new_model.add(Dense(2, activation = 'softmax', input_dim=2048))
```

In the last line, we use 2 because we are doing to retrain a new model to learn to differentiate cats and dogs -- so we only have 2 image classes. As noted before, the bottleneck output is of size 2048, so this is our input_dim to the `Dense` layer. Lastly, we insert a softmax activation to ensure our image class outputs can be interpreted as probabilities. 

To finish, we just need a few more standard tensorflow steps:

```
# For a binary classification problem
new_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
new_model.fit(processed_imgs_array, one_hot_labels, epochs=2, batch_size=32)
```

Where `processed_imgs_array` is an array of size (number_images_in_training_set, 224,224,3). `labels` is a Python list of the ground truth image classes. These are scalars corresponding to the class of the image in the training data. `num_classes=2`, so `labels` is just a list of length `number_of_images_in_training_set` containing 0’s and 1’s.

To recap, you need 3 ingredients to use transfer learning:
A pretrained model
Access to data
You need inputs to be “similar enough” to inputs of pre-trained model. Similar enough means that the inputs must be of the same format (e.g. shape of input tensors, data types…) and of similar interpretation. For example, if you are using a model pretrained for image classification, images will work as input! However, some clever folk have formatted audio to run through a pretrained image classifier, with some cool results. So, as ever, fortune favors the creative. 
Training labels.

We skipped over the ‘access to data’ bit above. Check out a full working example below for a demonstration of one strategy for locally loading files.


