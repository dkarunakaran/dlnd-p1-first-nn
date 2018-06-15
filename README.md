```
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # 1X1 connvolution of the layer 7
    conv_1x1_7th_layer = tf.layers.conv2d(vgg_layer7_out,num_classes, 1,padding = 'same',
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                     name='conv_1x1_7th_layer')
    # Upsampling x 4
    upsampling1 = tf.layers.conv2d_transpose(conv_1x1_7th_layer,
                                                num_classes,
                                                4,
                                                strides= (2, 2),
                                                padding= 'same',
                                                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                                name='upsampling1')
    # 1X1 convolution of the layer 4
    conv_1x1_4th_layer = tf.layers.conv2d(vgg_layer4_out,
                                     num_classes,
                                     1,
                                     padding = 'same',
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                     name='conv_1x1_4th_layer')
    skip1 = tf.add(conv_1x1_4th_layer, upsampling1, name="skip1")

    # Upsampling x 4
    upsampling2 = tf.layers.conv2d_transpose(skip1,
                                    num_classes,
                                    4,
                                    strides= (2, 2),
                                    padding= 'same',
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                    name='upsampling2')

    # 1X1 convolution of the layer 3
    conv_1x1_3th_layer = tf.layers.conv2d(vgg_layer3_out,
                                     num_classes,
                                     1,
                                     padding = 'same',
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                     name='conv_1x1_3th_layer')
    skip2 = tf.add(conv_1x1_3th_layer, upsampling2, name="skip2")

    # Upsampling x 8.
    upsampling3 = tf.layers.conv2d_transpose(skip2, num_classes, 16,
                                                  strides= (8, 8),
                                                  padding= 'same',
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                                  name='upsampling3')


    return upsampling3
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

