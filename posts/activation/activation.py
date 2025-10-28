import numpy as np
import matplotlib.pyplot as plt

def print_activation_function(name, cls, x_range=(-10, 10), num_points=100):
    x = [i * (x_range[1] - x_range[0]) / num_points + x_range[0] for i in range(num_points + 1)]
    y1 = [cls.forward(xi) for xi in x]
    y2 = [cls.backward(xi) for xi in x]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, y1)
    plt.title(f'Activation Function: {name}')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(x, y2)
    plt.title(f'Activation Function Derivative: {name}')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.savefig(f'content/posts/activation/activation_{name.lower()}.png')

class relu:
    def __init__(self):
        pass

    def forward(self, x):
        return max(0, x)

    def backward(self, x):
        return 1 if x > 0 else 0
    
class gelu:
    def __init__(self):
        pass

    def forward(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))))
    
    def backward(self, x):
        a = (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))))
        b = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3)))
        c = np.sqrt(2 / np.pi) * (1 + 0.044715 * 3 * (x ** 2))
        return 0.5 * a + 0.5 * x * (1 - (b ** 2)) * c
    
class swish:
    def __init__(self):
        pass

    def forward(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return x * sigmoid
    
    def backward(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return ((1 + np.exp(-x)) + x * np.exp(-x)) * (sigmoid ** 2)
    
if __name__ == "__main__":
    # ReLU Activation Function
    relu_activation = relu()
    print_activation_function("ReLU", relu_activation)

    # GELU Activation Function
    gelu_activation = gelu()
    print_activation_function("GELU", gelu_activation)

    ## Swish Activation Function
    swish_activation = swish()
    print_activation_function("Swish", swish_activation)