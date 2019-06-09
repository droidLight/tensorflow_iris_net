import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

print(tf.__version__)
data = pd.read_csv('Iris.csv')
train_split = int(data.shape[0]*0.7)
del data['Id']

#one hot encode labels
y = data['Species']
del data['Species']
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = y.reshape(-1,1)
onehotencoder = OneHotEncoder(categorical_features=[0])
y = onehotencoder.fit_transform(y).toarray()


#train test split
x = data.values
x, y = shuffle(x, y)
train_x = x[0:train_split,:]
train_y = y[0:train_split,:]
test_x = x[train_split:x.shape[0],:]
test_y = y[train_split:y.shape[0],:]

#neural network

#neurons in each layer
input_layer_num = train_x.shape[1]
hidden_layer_num = 100
output_layer_num = 3

#placeholder
x = tf.placeholder(tf.float32, [None, input_layer_num])
y = tf.placeholder(tf.float32, [None, output_layer_num])

#hyper parameters
epoch = 200
learning_rate = 0.01
batch_size = 32

#weights and biases
with tf.name_scope("weights_and_bias") as scope: 
    weights = {
        'hidden_layer':tf.Variable(tf.random_normal([input_layer_num, hidden_layer_num])),
        'output_layer':tf.Variable(tf.random_normal([hidden_layer_num, output_layer_num]))
    }

    bias = {
        'hidden_layer':tf.Variable(tf.random_normal([hidden_layer_num])),
        'output_layer':tf.Variable(tf.random_normal([output_layer_num]))
    }

#computation graph
with tf.name_scope("hidden_and_output_layer") as scope:
    hidden_layer = tf.add(tf.matmul(x, weights['hidden_layer']), bias['hidden_layer'])
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.add(tf.matmul(hidden_layer, weights['output_layer']), bias['output_layer'])

#calculate cost function
with tf.name_scope("cost_function") as scope:
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = y))

#optimizer to minimize loss
with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_function)



#intializing all variables
init = tf.initialize_all_variables()

#creating session and training the neural net
sess = tf.Session()
sess.run(init)

for i in range(epoch):
    #fit and train the model
    sess.run(optimizer, feed_dict={x:train_x, y:train_y})
    #compute cost
    cost = sess.run(cost_function, feed_dict={x:train_x, y:train_y})
    
    print("Epoch: "+str(i+1)+"  cost: "+str(cost))

print("Training complete")

#predictions
predictions = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
print("Accuracy:", accuracy.eval({x: test_x, y: test_y}, session = sess))

