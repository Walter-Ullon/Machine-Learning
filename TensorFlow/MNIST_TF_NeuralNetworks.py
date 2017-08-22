# MNIST dataset with TensorFlow and Neural Networks.
# Three hidden layers, 256 nodes a piece.

# Import packages:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# ------------ Load data into variable 'MNIST'. Inspect: -----------------
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Print variable type:
print("1*-----------------------------")
print(type(mnist))

# Inspect variable (55,000 images, each is an array of 784 units (28 by 28 pixels))
# Note: mnist.train -> training data, 55k points / mnist.test, 10k points / 
# mnist.validation, 5k points.

print("2*-----------------------------")
print(mnist.train.images.shape)

# Pull one sample and visualize it as an image reshape as square matrix.
sample = mnist.train.images[2].reshape(28,28) 
# print(mnist.train.images[4]) # vectorization of image.
print("3*-----------------------------")
print(mnist.train.labels[2])   # image label in  vector form.
plt.imshow(sample, cmap='Greys')
plt.show()


# ------------ Define learning parameters: -----------------
# How quickly we adjust the 'cost' function
learning_rate = 0.001

# Training cycles - fifteen training cycles in batches of 100:
training_epochs = 50
batch_size = 100

n_classes = 10 # ten classes of numbers (0-9).
n_samples = mnist.train.num_examples #total number of samples (55k)

n_input = 784

# Number of neurons on the hidden layers.
n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 256


# ------------ Define perceptron: -----------------

def multilayer_perceptron(x, weights, biases):
    '''
    x: placeholder for data input.
    weights: dictionary of weights.
    biases: dictionary of biases.
    '''
    
    # First hidden layer w/ RELU:
    # X*W +B
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # RELU(X*W + B) -> 0 if x < 0, x otherwise.
    layer_1 = tf.nn.relu(layer_1)

    # Second hidden layer w/ RELU:
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
     # Third hidden layer w/ RELU:
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Output hidden layer w/ RELU:
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer


# ------------ Create dictionary of weights, biases: -----------------
weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), # [7884, 256]...
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), #[256, 256]
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])), #[256, 256]
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes])) #[256, 10]
        }

biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
            
        }


# ------------ Set placeholders: -----------------
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])


# ------------ Set model: -----------------
pred = multilayer_perceptron(x, weights, biases)


# ------------ Set cost optimization and Optimizer: -----------------
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# ------------ Train model: -----------------
# grab 100 samples from trining data at a time, in tuples (matrix, labels).
t = mnist.train.next_batch(batch_size)


# ------------ Run session: -----------------
# Set session object:
sess = tf.InteractiveSession()

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

print()
print("4*-----------------------------")
for epoch in range(training_epochs):
    # Cost
    avg_cost = 0.0
    
    total_batch = int(n_samples/batch_size)
    
    for i in range(total_batch):
        
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
        
        avg_cost += c/total_batch
    
    print('Epoch: {} cost {:.4f}'.format(epoch+1, avg_cost))

print('Model has completed {} Epochs of training'.format(training_epochs))



# ------------ Evaluation of model: -----------------
# Return vector of True/False when matching prediction to actual.
correct_predictions = tf.equal(tf.arg_max(pred,1), tf.arg_max(y,1))

# Cast True/False values as floats
correct_predictions = tf.cast(correct_predictions, 'float')

# Get the mean of total predicitons.
accuracy = tf.reduce_mean(correct_predictions)
print("Accuracy: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))









