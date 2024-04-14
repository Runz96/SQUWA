import torch

def calculate_centroid(z_cluster):
    return torch.mean(z_cluster, dim=0)

def calculate_intra_distance(z_cluster, centroid_cluster):
    diff = z_cluster - centroid_cluster
    dist = torch.sqrt(torch.sum(diff**2, dim=1))
    return torch.mean(dist)

def calculate_inter_distance(centroid_1, centroid_2):
    diff = centroid_1 - centroid_2
    dist = torch.sqrt(torch.sum(diff**2))
    return dist

def calculate_cmc_loss(cluster_vectors):
    num_clusters = len(cluster_vectors)
    intra_distances = []
    inter_distances = []

    # Calculate intra-cluster distances
    centroids = [calculate_centroid(z_cluster) for z_cluster in cluster_vectors]
    for i in range(num_clusters):
        intra_dist = calculate_intra_distance(cluster_vectors[i], centroids[i])
        intra_distances.append(intra_dist)

    # Calculate inter-cluster distances
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            inter_dist = calculate_inter_distance(centroids[i], centroids[j])
            inter_distances.append(inter_dist)

    # Combine intra and inter cluster distances to calculate the loss
    # This is a basic example, you might want to modify the formula as per your requirement
    if len(inter_distances) > 0:
        # loss = sum(intra_distances) / sum(inter_distances)
        # loss = sum(intra_distances) / len(intra_distances)
        loss_intra = torch.log(sum(intra_distances) + 1e-9)
        loss_inter = torch.log(sum(inter_distances) + 1e-9)
        # loss = loss_intra, loss_inter
    else:
        loss = torch.tensor(0.0)  # Handle the case where there are no inter-cluster distances (e.g., only one cluster)

    return loss_intra, loss_inter