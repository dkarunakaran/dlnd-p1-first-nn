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

```
import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)

```
