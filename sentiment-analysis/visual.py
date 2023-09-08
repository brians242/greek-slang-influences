import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

def affinity_propagation(X, max_iter=200, damping=0.5, convergence_iter=15, random_state=None):
    affinity_prop = AffinityPropagation(max_iter=max_iter, damping=damping, convergence_iter=convergence_iter, random_state=random_state)
    cluster_centers_indices = affinity_prop.fit_predict(X)
    cluster_centers = np.array([X[cluster_centers_indices == i].mean(axis=0) for i in np.unique(cluster_centers_indices)])
    return cluster_centers_indices, cluster_centers

def mahalanobis_distance(query_point, clusters, covariances, regularization=1e-6):
    distances = []
    for center, covariance in zip(clusters, covariances):
        inv_covariance = np.linalg.pinv(covariance + regularization * np.eye(covariance.shape[0]))
        distance = mahalanobis(query_point, center, inv_covariance)
        distances.append(distance)
    return distances

def compute_covariance_matrix(cluster_data):
    if cluster_data.shape[0] > 1:
        return np.cov(cluster_data, rowvar=False)
    else:
        num_features = cluster_data.shape[1]
        return np.eye(num_features)

def plot_clusters(X, cluster_centers_indices, cluster_centers, query_point, min_cluster_index):
    num_clusters = len(np.unique(cluster_centers_indices))
    cmap = plt.get_cmap('tab20', num_clusters)

    plt.scatter(X[:, 0], X[:, 1], c=cluster_centers_indices, cmap=cmap)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', color='black', label='Cluster Centers')
    plt.scatter(query_point[0], query_point[1], marker='D', color='red', s=100, label='Query Point')
    if min_cluster_index != -1:
        min_cluster_center = cluster_centers[min_cluster_index]
        plt.scatter(min_cluster_center[0], min_cluster_center[1], marker='o', color='green', s=100, label='Closest Cluster')
    plt.legend()
    plt.title("Affinity Propagation Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def embed_linguistic_data(linguistic_data):
    model = Word2Vec(sentences=linguistic_data, vector_size=100, window=5, min_count=1, sg=0)
    embeddings = [model.wv[word] for sentence in linguistic_data for word in sentence]
    return np.array(embeddings)

if __name__ == "__main__":
    linguistic_data = [
    ['The', 'cat', 'walked', 'across', 'the', 'street'],
    ['Children', 'played', 'in', 'the', 'park'],
    ['Books', 'open', 'doors', 'to', 'new', 'worlds'],
    ['Laughter', 'is', 'contagious', 'and', 'uplifting'],
    ['Time', 'flies', 'when', 'you', 'are', 'having', 'fun'],
    ['Music', 'soothes', 'the', 'soul', 'and', 'calms', 'the', 'mind'],
    ['The', 'ocean', 'is', 'vast', 'and', 'mysterious'],
    ['Dreams', 'hold', 'the', 'potential', 'for', 'unlimited', 'imagination'],
    ['Learning', 'is', 'a', 'lifelong', 'journey', 'of', 'discovery'],
    ['Nature', 'is', 'full', 'of', 'wonders', 'to', 'behold'],
    ['Kindness', 'can', 'brighten', 'even', 'the', 'darkest', 'days'],
    ['Exploration', 'leads', 'to', 'innovation', 'and', 'progress'],
    ['Colors', 'paint', 'the', 'world', 'with', 'beauty', 'and', 'vibrancy'],
    ['Adventure', 'awaits', 'those', 'who', 'dare', 'to', 'explore'],
    ['Imagination', 'knows', 'no', 'boundaries', 'or', 'limitations']
]
    embeddings = embed_linguistic_data(linguistic_data)

    n_dimensions = 2
    perplexity = 30
    tsne_model = TSNE(n_components=n_dimensions, perplexity=perplexity)
    n_dimensional_points = tsne_model.fit_transform(embeddings)

    mean_values = np.mean(n_dimensional_points, axis=0)
    std_values = np.std(n_dimensional_points, axis=0)
    z_score_normalized_data = (n_dimensional_points - mean_values) / std_values

    print("Mean of z-score normalized data:", np.mean(z_score_normalized_data, axis=0))
    print("Standard deviation of z-score normalized data:", np.std(z_score_normalized_data, axis=0))

    plt.scatter(z_score_normalized_data[:, 0], z_score_normalized_data[:, 1], marker='o', cmap='viridis')
    plt.xlabel('Dimension 1: Normalized')
    plt.ylabel('Dimension 2: Normalized')
    plt.title('2D Scatter Plot')
    plt.show()

    np.random.seed(42)
    X = z_score_normalized_data

    cluster_centers_indices, cluster_centers = affinity_propagation(X)

    if not cluster_centers.any():
        print("Clustering failed to converge.")
    else:
        clusters_data = [X[cluster_centers_indices == i] for i in range(len(cluster_centers))]
        covariances = [compute_covariance_matrix(cluster_data) for cluster_data in clusters_data]

        query_point = np.array([0.0, 0.0])  

        distances = mahalanobis_distance(query_point, cluster_centers, covariances)

        min_distance = float('inf')
        min_cluster_index = -1
        for i, distance in enumerate(distances):
            if distance < min_distance:
                min_distance = distance
                min_cluster_index = i

        plot_clusters(X, cluster_centers_indices, cluster_centers, query_point, min_cluster_index)

        for i, distance in enumerate(distances):
            print(f"Cluster {i+1} - Mahalanobis Distance: {distance:.4f}")

        if min_cluster_index != -1:
            print(f"\nCluster with the smallest Mahalanobis distance: Cluster {min_cluster_index + 1}")