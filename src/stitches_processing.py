import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance

def prune_points(points, threshold):
    if len(points) == 1: return points

    clustering = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None).fit(points[:,0].reshape(-1,1))

    
    labels = np.unique(clustering.labels_)      # get the unique labels from the clustering result

    if len(labels) == 1:                        # only one label - return first point from cluster (the closest one to the other side of the image)
        pruned_points = [points[0,:]]
        return np.array(pruned_points)

    pruned_points = []

    # find centroids of clusters
    centroids = []
    for label in labels:
        cluster_indices = np.where(clustering.labels_ == label)[0]
        cluster_points = points[cluster_indices]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    for i, label in enumerate(labels):
        
        cluster_indices = np.where(clustering.labels_ == label)[0]                      # indices of points in the current cluster
        
        cids = np.ones(len(labels), dtype=bool)                                         # cluster indices
        cids[i] = False
        
        distances = distance.cdist(points[cluster_indices], centroids[cids,:])          # calculate the distances from each point in the cluster to other centroids
        closest_point_index = cluster_indices[np.argmin(distances)//(len(labels)-1)]    # index of the point closest to a centroid
        pruned_points.append(points[closest_point_index])

    pruned_points = np.array(pruned_points)
    return pruned_points

def process_st_candidates(cnd_x, cnd_y, masks, th):
        masks_indices = []
        for mask in masks:
                indices = np.argwhere(mask)
                indices_list = [tuple(index) for index in indices]
                masks_indices.append(indices_list)
        
        out = ([],[])
        out_x = [[],[]]
        out_y = [[],[]]
        for i in range(len(cnd_x)):
                for x, y in zip(cnd_x[i], cnd_y[i]):
                        if (y,x) not in masks_indices[i]:
                                continue
                        out[i].append((x,y))
                # print(out[i])
                if len(out[i])==0: continue
                if i == 0: out[i].reverse()
                pruned = prune_points(np.array(out[i]), th)
                out_x[i] = list(pruned[:,0])
                out_y[i] = list(pruned[:,1])
                
                        

        return out_x, out_y

def do_stitching(x, y, centers, th):
        if len(x[1])==0: return []
        points_A = np.hstack((np.array(x[0]).reshape(len(x[0]), 1), np.array(y[0]).reshape(len(y[0]), 1)))
        points_B = np.hstack((np.array(x[1]).reshape(len(x[1]), 1), np.array(y[1]).reshape(len(y[1]), 1)))
        points_A = points_A + centers[0,:]
        points_B = points_B + centers[1,:]

        stitches = []
        for point_A in points_A:
                distances = np.abs(points_B[:,0] - point_A[0])
                if distances.min() > th:
                        continue
                point_B = points_B[np.argmin(distances)]
                stitches.append((tuple(point_A), tuple(point_B)))

        return stitches