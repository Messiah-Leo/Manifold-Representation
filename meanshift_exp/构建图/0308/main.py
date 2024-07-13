from sklearn.datasets import load_digits
from sklearn.manifold import MDS

X, _ = load_digits(return_X_y=True)
embedding = MDS(n_components=2, normalized_stress='auto')
X_transformed = embedding.fit_transform(X[:100])