# Online Python - IDE, Editor, Compiler, Interpreter
import numpy as np

# Fake embeddings (e.g., latent vectors of size 3)
embeddings_normal = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])

embeddings_anomaly = np.array([
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5],
    [1.6, 1.7, 1.8]
])

# Labels
y_normal = np.array([0, 0])
y_anomaly = np.array([1, 1, 1])

# Stack
X = np.vstack((embeddings_normal, embeddings_anomaly))
y = np.concatenate((y_normal, y_anomaly))

print("X:\n", X)
print("y:\n", y)

print("Shapes:", X.shape, y.shape)

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
aucs = []

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"\n=== Fold {i} ===")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Train indices:", train_index)
    print("Test indices:", test_index)

    print("y_train:", y_train)
    print("y_test:", y_test)

    print("X_train:\n", X_train)
    print("X_test:\n", X_test)

    # Optional: check class balance
    print("Train class distribution:", np.bincount(y_train))
    print("Test class distribution:", np.bincount(y_test))