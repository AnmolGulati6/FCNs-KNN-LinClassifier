# FCNs-KNN-LinClassifier
A collection of my machine learning algorithm implementations in Python, including fully connected neural networks, linear classifiers, and k-Nearest Neighbors.

## Contents

1. `fully_connected_networks.py` - Implementation of a fully connected neural network.
2. `linear_classifier.py` - Implementation of a linear classifier.
3. `knn.py` - Implementation of the k-Nearest Neighbors algorithm.

## Installation

Clone the repository:
```bash
git clone https://github.com/AnmolGulati6/FCNs-KNN-LinClassifier.git

## Usage

### Fully Connected Neural Networks

The `fully_connected_networks.py` file contains classes to build and train a simple fully connected neural network.

```python
from fully_connected_networks import FullyConnectedLayer, NeuralNetwork

# Example usage
network = NeuralNetwork()
network.add_layer(FullyConnectedLayer(2, 3))
network.add_layer(FullyConnectedLayer(3, 1))
network.train(x_train, y_train, epochs=100, learning_rate=0.01)
```

### Linear Classifier

The `linear_classifier.py` file contains a class for a linear classifier with training functionality.

```python
from linear_classifier import LinearClassifier

# Example usage
classifier = LinearClassifier(input_dim=2, output_dim=1)
classifier.train(x_train, y_train, epochs=100, learning_rate=0.01)
```

### k-Nearest Neighbors

The `knn.py` file contains a class for the k-Nearest Neighbors algorithm.

```python
from knn import KNearestNeighbors

# Example usage
knn = KNearestNeighbors(k=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

Feel free to customize the README and other details as per your requirements.
