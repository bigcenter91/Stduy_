import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the diabetes dataset
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

# Reshape y to have shape (n_samples, 1)
y = y.reshape(-1, 1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=337)

# Convert the input data to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Define the placeholders for input features and target
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Define the variables for weights and bias with dtype=tf.float32
w = tf.compat.v1.Variable(tf.random.normal([10, 1], dtype=tf.float32), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1], dtype=tf.float32), name='bias')

# Define the model
hypothesis = tf.compat.v1.matmul(x, w) + b

# Define the loss function
loss = tf.reduce_mean(tf.square(hypothesis - y))

# Define the optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# Create a session and initialize variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Training
epochs = 3000
for epoch in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss_val)

# Evaluation
y_train_pred = sess.run(hypothesis, feed_dict={x: x_train})
y_test_pred = sess.run(hypothesis, feed_dict={x: x_test})

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R2 Score (Train):", r2_train)
print("R2 Score (Test):", r2_test)

# Close the session
sess.close()