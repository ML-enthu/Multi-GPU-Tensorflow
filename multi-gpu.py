import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import os
import time

print(tf.VERSION)
#print(np.__version__)
print(tf.keras.__version__)


class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []
    def before_run(self, run_context):
        self.iter_time_start = time.time()
    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)

def input_fn(images, labels, epochs, batch_size):
    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 5000
    ds = ds.apply(tf.contrib.data.shuffle_and_repeat(SHUFFLE_SIZE, count=epochs)).batch(batch_size)
    #ds = ds.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    # Return the dataset. (L)
    return ds

def eval_input_fn(images, labels, batch_size):
    # Convert the inputs to a Dataset. (E)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    assert batch_size is not None, "batch_size must not be None"
    # Batch the examples. (T)
    dataset = dataset.batch(batch_size)
    # Return the dataset. (L)
    return dataset

###################################### Pre-processing Step #####################################
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)


train_images = np.asarray(train_images, dtype=np.float32) / 255
# Convert the train images and add channels
train_images = train_images.reshape((TRAINING_SIZE, 28, 28, 1))
test_images = np.asarray(test_images, dtype=np.float32) / 255
# Convert the test images and add channels
test_images = test_images.reshape((TEST_SIZE, 28, 28, 1))


# Categories we are predicting from (0-9)
LABEL_DIMENSIONS = 10
train_labels = tf.keras.utils.to_categorical(train_labels, 
                                             LABEL_DIMENSIONS)
test_labels = tf.keras.utils.to_categorical(test_labels,
                                            LABEL_DIMENSIONS)
# Cast the labels to floats
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)


################################ Model definition ###########################################
 
inputs = tf.keras.Input(shape=(28,28,1))  	
x = tf.keras.layers.Conv2D(filters=32, 
                           kernel_size=(3, 3), 
                           activation=tf.nn.relu)(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = tf.keras.layers.Conv2D(filters=64, 
                           kernel_size=(3, 3), 
                           activation=tf.nn.relu)(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = tf.keras.layers.Conv2D(filters=64, 
                           kernel_size=(3, 3), 
                           activation=tf.nn.relu)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
predictions = tf.keras.layers.Dense(LABEL_DIMENSIONS,
                                    activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)


############################## Model Compilation ##########################################



optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


#################################### code for Mutli- GPU training ####################################
################################## Distribution Strategy ######################################
NUM_GPUS = 2
strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
config = tf.estimator.RunConfig(train_distribute=strategy)
estimator = tf.keras.estimator.model_to_estimator(model, config=config)


########################################################################
time_hist = TimeHistory()
BATCH_SIZE = 128
EPOCHS = 10


########################################################################
## Training on multiple GPUs

#import logging
#logging.getLogger().setLevel(logging.INFO)
print("\n")
print(" Starting Training !")
print("\n")
estimator.train(lambda:input_fn(train_images,
                                train_labels,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE),
                hooks=[time_hist])

print('\n\n\n')
print("*"*80)
print('\n')

total_time = sum(time_hist.times)
print("Training Done!")
print("\n")
print("total time with {} GPU(s): {} seconds".format(NUM_GPUS, total_time))
avg_time_per_batch = np.mean(time_hist.times)
print("{} images/second with {} GPU(s)".format(BATCH_SIZE*NUM_GPUS/avg_time_per_batch, NUM_GPUS))
print('\n')


########################################################################
## Evaluating the results
evaluate_result = estimator.evaluate(input_fn=lambda: eval_input_fn(test_images, test_labels, BATCH_SIZE))

print('\n')
print ("Evaluation results")
for key in evaluate_result:
   print("   {}, was: {}".format(key, evaluate_result[key]))
