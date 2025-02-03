import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


data = np.array([
    [5.3, 3.7, 'Setosa'],
    [5.1, 3.8, 'Setosa'],
    [7.2, 3.0, 'Virginica'],
    [5.4, 3.4, 'Setosa'],
    [5.1, 3.3, 'Setosa'],
    [5.4, 3.9, 'Setosa'],
    [7.4, 2.8, 'Virginica'],
    [6.1, 2.8, 'Versicolor'],
    [7.3, 2.9, 'Virginica'],
    [6.0, 2.7, 'Versicolor'],
    [5.8, 2.8, 'Virginica'],
    [6.3, 2.3, 'Versicolor'],
    [5.1, 2.5, 'Versicolor'],
    [6.3, 2.5, 'Versicolor'],
    [5.5, 2.4, 'Versicolor']
], dtype=object)


X_train = np.array(data[:, :2], dtype=float)  
y_train = np.array(data[:, 2]) 


X_test = np.array([5.2, 3.1])


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


distances = []
for i in range(len(X_train)):
    dist = euclidean_distance(X_test, X_train[i])
    distances.append((dist, y_train[i], X_train[i]))  

k = 3
distances.sort()  
nearest_neighbors = distances[:k]


neighbor_classes = [neighbor[1] for neighbor in nearest_neighbors]


predicted_class = Counter(neighbor_classes).most_common(1)[0][0]


print(f"Predicted species for Sepal Length = {X_test[0]}, Sepal Width = {X_test[1]}: {predicted_class}")


plt.figure(figsize=(6, 6))

species_colors = {'Setosa': 'blue', 'Versicolor': 'green', 'Virginica': 'purple'}
for i, label in enumerate(y_train):
    plt.scatter(X_train[i, 0], X_train[i, 1], color=species_colors[label], edgecolors='black', s=100, 
                label=f"{label}" if label not in plt.gca().get_legend_handles_labels()[1] else "")

plt.scatter(X_test[0], X_test[1], color='red', marker='*', s=200, label="New Data")


for neighbor in nearest_neighbors:
    plt.plot([X_test[0], neighbor[2][0]], [X_test[1], neighbor[2][1]], 'r--', alpha=0.6)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Manual KNN Classification")
plt.legend()
plt.grid()
plt.show()
