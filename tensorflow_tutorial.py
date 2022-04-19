import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Interpreter = Python 3.9, not Anaconda3

# alternative to the mnist dataset - slightly more challenging problem than regular mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# the dataset doesn't have any class names - only labels from 0-9 - so these have to be specified
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# shows the first training image as a sort of heatmap
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# normalises values to be between 0 and 1
train_images = train_images / 255.0

test_images = test_images / 255.0

# plots the first 25 training images and their classifications
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# set up the network - tf.keras is a high-level API to build and train models in TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # transforms the 28x28 2D array into a 784 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # standard dense layer with 128 nodes
    tf.keras.layers.Dense(10)  # standard dense layer with 10 nodes - output layer
])

# the model's compile step - adds an optimiser, loss function
model.compile(optimizer='adam',  # optimiser - how the model is updated based on the data it sees and the loss func.
              # Adam = SGD based on adaptive estimation of first- and second-order moments
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # loss function - here, crossentropy loss
              metrics=['accuracy'])  # metrics - how the training and testing steps are monitored

# fit the model
model.fit(train_images, train_labels, epochs=10)

# evaluate - on the test datasets
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)  # accuracy on test is less than that on training - represents overfitting

# add a softmax layer to convert logits to more readily interpretable probabilities - ready for making predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# make predictions on all the test images
predictions = probability_model.predict(test_images)


# access the prediction on the first test image and retrieve in which classification the model has the highest confidence
predictions[0]
if np.argmax(predictions[0]) == test_labels[0]:
    print('Prediction correct')


# function to plot the image, its predicted label and whether the prediction is correct
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


# function to plot how confident the model is in each prediction, with a correct prediction coloured blue, and a false one red
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# tf.keras models are optimised to make predictions on a batch (collection) of examples - organise in a list by
# ordering them in another dimension
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])

print(img.shape)