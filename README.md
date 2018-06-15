```
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # Reshape the label same as logits 
    label_reshaped = tf.reshape(correct_label, (-1,num_classes))

    # Converting the 4D tensor to 2D tensor. logits is now a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_reshaped))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1e-3
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    train_op = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss)    
    
    return logits, train_op, loss
```

# dlnd-p1-first-nn
Deep learning nano degreee first project


## Instructions
1. Clone the repo

2. Change the directory 
```
    cd dlnd-p1-first-nn
```
3. Download anaconda or miniconda based on the instructions in the Anaconda lesson.

4. Create a new conda environment:
```
    conda create --name dlnd python=3
```
5. Enter your new environment:
```
    Mac/Linux: >> source activate dlnd
    Windows: >> activate dlnd
```
6. Ensure you have numpy, matplotlib, pandas, and jupyter notebook installed by doing the following:
```
    conda install numpy matplotlib pandas jupyter notebook
```

7. Run the following to open up the notebook server:
```
    jupyter notebook
```
8. In your browser, open dlnd-your-first-neural-network.ipynb

9. Follow the instructions in the notebook will lead you through the project.

