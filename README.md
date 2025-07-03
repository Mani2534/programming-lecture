# programming-lecture

This project demonstrates the practical application of supervised learning using a simple feedforward neural network on the MNIST dataset. MNIST consists of 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels, widely used as a benchmark for image classification.
 Model Theory
The architecture used in both implementations is a Multi-Layer Perceptron (MLP), a foundational type of artificial neural network. It consists of:

Input Layer: Flattens the 28×28 pixel image into a 784-element vector.

Hidden Layer: Contains 64 fully connected neurons with ReLU (Rectified Linear Unit) activation to introduce non-linearity.

Output Layer: A dense layer with 10 neurons—one for each digit class.

TensorFlow: Uses softmax activation to convert outputs to probability distribution.

PyTorch: Outputs raw logits; softmax is applied internally during loss computation (CrossEntropyLoss).

Learning Objective

The goal of training is to minimize the classification error on unseen data by adjusting the weights and biases of the network through backpropagation and gradient descent (using the Adam optimizer). The loss function measures how far the predicted outputs are from the true labels.

TensorFlow uses CategoricalCrossentropy with one-hot encoded labels.
PyTorch uses CrossEntropyLoss with integer-encoded class labels.

Why Compare TensorFlow and PyTorch?

Both frameworks are widely used for deep learning but offer different philosophies:

TensorFlow emphasizes production-ready deployment, with model export tools like TensorFlow Lite.
PyTorch is often favored for research due to its dynamic computation graph and native Pythonic interface.

This project provides a side-by-side comparison of how the same model behaves in both frameworks with respect to:

Development experience
Training performance
Inference speed
Exportability for deployment (TFLite vs ONNX)
