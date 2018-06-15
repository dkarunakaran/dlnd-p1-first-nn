```
# Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        print("Epoch {}".format(epoch + 1))
        training_loss = 0
        training_samples_length = 0
        for image, label in get_batches_fn(batch_size):
            training_samples_length += len(image)
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: label,
                keep_prob: 0.5,
                learning_rate: 0.0001
            })
            training_loss += loss
            print(loss)
        
        # Total training loss
        training_loss /= training_samples_length
        print("********************Total loss***********************")
        print(training_loss)
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

