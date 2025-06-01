# mall_customers_clustering.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def load_data(filepath):
    """
    Load the Mall Customers Dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def plot_elbow_method(data, max_clusters=10):
    """
    Use the Elbow Method to find the optimal number of clusters.
    """
    wcss = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_clusters + 1), wcss, 'bo-', markersize=8)
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid(True)
    plt.show()

def apply_kmeans(data, n_clusters):
    """
    Apply KMeans clustering algorithm.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def plot_clusters(df, clusters):
    """
    Plot clusters with respect to Annual Income and Spending Score.
    """
    df_plot = df.copy()
    df_plot['Cluster'] = clusters
    plt.figure(figsize=(10,7))
    sns.scatterplot(data=df_plot, x='Annual Income (k$)', y='Spending Score (1-100)', 
                    hue='Cluster', palette='Set2', s=100, alpha=0.7)
    plt.title('Customer Segments')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Cluster')
    plt.show()

def main():
    # Load data
    df = load_data('Mall_Customers.csv')
    
    # Select features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Plot Elbow method to find optimal k
    plot_elbow_method(X, max_clusters=10)
    
    # Based on elbow plot, select optimal clusters (example: 5)
    optimal_k = 5
    
    # Apply KMeans clustering
    clusters, model = apply_kmeans(X, optimal_k)
    
    # Plot the clusters
    plot_clusters(df, clusters)
    
    # Optional: save the dataframe with cluster labels
    df['Cluster'] = clusters
    df.to_csv('Mall_Customers_with_Clusters.csv', index=False)

if __name__ == '__main__':
    main()
