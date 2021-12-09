"""Train the model"""

import logging
import os
import pathlib

import tensorflow as tf

from model.model_fn import ViT
from model.utils import set_logger


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return tf.losses.CategoricalCrossentropy(from_logits=True)(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(123)

    # Set the logger
    cwd = os.getcwd()
    set_logger(os.path.join(cwd, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # For shorter training time, We'll use caltech101 instead of imagenet used in the paper
    data_dir = pathlib.Path(r'C:\Users\K\tensorflow_datasets\caltech101')

    batch_size = 32
    img_height = 256
    img_width = 256

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                       label_mode='categorical',
                                                       validation_split=0.2,
                                                       subset="training",
                                                       seed=123,
                                                       image_size=(img_height, img_width),
                                                       batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                     label_mode='categorical',
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=123,
                                                     image_size=(img_height, img_width),
                                                     batch_size=batch_size)

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Define the model
    logging.info("Creating the model...")
    model = ViT(50, 100, 10, 0.1, 3, 32, 102)

    # Train the model
    num_epochs = 3
    logging.info("Starting training for {} epoch(s)".format(num_epochs))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for epoch in range(num_epochs):
        epoch_loss =tf.losses.CategoricalCrossentropy(from_logits=True)
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_ds:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_accuracy.update_state(y, model(x, training=True))

        print("Epoch {:03d}: Accuracy: {:.3%}".format(epoch, epoch_accuracy.result()))

    logging.info("End of training for {} epoch(s)".format(params.num_epochs))