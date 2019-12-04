import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def find_neighbor_points(feature_matrix, epsilon, point):
    neighbours = []
    for other_points in range(0, feature_matrix.shape[0]):
        feature_difference = feature_matrix[point] - feature_matrix[other_points]
        for j in range(len(feature_difference)):
            total_sum = 0
            square_root = 0
            feature_difference[j] = feature_difference[j] * feature_difference[j]
        total_sum = sum(feature_difference)
        square_root = np.sqrt(total_sum)

        if square_root <= epsilon:
            neighbours.append(other_points)
    return neighbours


def redefine_cluster(point, neighbor_points, cluster, eps, minpts, label_list, feature_matrix):
    label_list[point] = cluster
    for pt in neighbor_points:
        if label_list[pt] == -1:
            label_list[pt] = cluster

        if label_list[pt] == 0:
            label_list[pt] = cluster
            neighbor = find_neighbor_points(feature_matrix, eps, pt)
            if len(neighbor) >= minpts:
                neighbor_points += neighbor


def dbscan(feature_matrix, epsilon, mpts, nf_name):
    cluster = 0
    size_of_label_list = feature_matrix.shape[0]
    label_list = [0 for i in range(size_of_label_list)]
    label_list = np.asarray(label_list)

    for point in range(0, feature_matrix.shape[0]):
        if label_list[point] != 0:
            continue

        neighbor_points = find_neighbor_points(feature_matrix, epsilon, point)
        if (len(neighbor_points)) < mpts:
            label_list[point] = -1
            continue
        cluster = cluster + 1
        redefine_cluster(point, neighbor_points, cluster, epsilon, mpts, label_list, feature_matrix)

    centroids = {}
    min_lable = int(np.min(label_list))
    max_lable = int(np.max(label_list))
    for label in range(min_lable, max_lable + 1):
        if label == 0:
            continue
        centroids[label] = np.asarray(np.where(label_list == label)) + 1

    f = open(nf_name + "_DBSCAN_result.txt", "w+")
    #     f.write(f'Total Iterations: {iterations}')
    #     f.write("\n\n")

    for (key, val) in centroids.items():
        f.write(f'cluster: {key}')
        f.write("\n\n")
        f.write(str(val))
        f.write("\n\n\n\n")
    f.close()
    print("Clustering Complete")
    print("Results written in " + nf_name + "_DBSCAN_result.txt file")
    #     print("Cluster: points in cluster = ")

    #     for key, value in centroids.items():
    #         print (str(key) + ":" + str(value))

    return label_list


def compute_similarity_coeff(label, ground_truth, nf_name):
    m11 = 0
    m10 = 0
    m01 = 0
    m00 = 0
    for i in range(0, len(ground_truth)):
        first_ground_truth = ground_truth[i];
        first_cluster_value = label[i];

        for j in range(0, len(ground_truth)):
            second_ground_truth = ground_truth[j]
            second_cluster_value = label[j]
            ground_truth_value = (first_ground_truth == second_ground_truth)
            cluster_similarity_value = (first_cluster_value == second_cluster_value)

            if (ground_truth_value == True and cluster_similarity_value == True):
                m11 += 1
            elif (ground_truth_value == True and cluster_similarity_value == False):
                m10 += 1
            elif (ground_truth_value == False and cluster_similarity_value == True):
                m01 += 1
            elif (ground_truth_value == False and cluster_similarity_value == False):
                m00 += 1

    jacard = float(m11 / (m11 + m01 + m10))
    #     print("jaccard_coefficient: ", jacard_coeff)
    rand = float((m11 + m00) / (m11 + m00 + m10 + m01))
    #     print("rand_index: ", rand_index)

    f = open(nf_name + "_DBSCAN_result.txt", "a+")
    f.write("\n\n")
    f.write(f'Jaccard Coefficient: {jacard}')
    f.write("\n\n")
    f.write(f'Rand Index: {rand}')


def plot_PCA_db(file, labels, plot_title):
    file = open(file, 'r');
    gene_data = file.readlines();
    gene_row_len = len(gene_data[0].split(' '))
    gene_data_matrix = [[np.float64(x) for x in line.split(' ')] for line in gene_data]
    gene_data_matrix = np.asarray(gene_data_matrix, dtype=float)
    data = gene_data_matrix[:, 2:gene_row_len]

    pca_df = pd.DataFrame(data, columns=['PC1', 'PC2'])
    pca_df["label"] = labels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    unique_list = []
    for x in labels:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    colors = ['r', 'g', 'b']
    for target, color in zip(unique_list, colors):
        indicesToKeep = pca_df['label'] == target
        ax.scatter(pca_df.loc[indicesToKeep, 'PC1']
                   , pca_df.loc[indicesToKeep, 'PC2']
                   , c=color
                   , s=50)
    ax.legend(unique_list)
    ax.set_title(plot_title)
    ax.grid()
    plt.show()


def plot_PCA(file, labels, plot_title):
    file = open(file, 'r');
    gene_data = file.readlines();
    gene_row_len = len(gene_data[0].split('\t'))
    gene_data_matrix = [[np.float64(x) for x in line.split('\t')] for line in gene_data]
    gene_data_matrix = np.asarray(gene_data_matrix, dtype=float)
    data = gene_data_matrix[:, 2:gene_row_len]
    pca = PCA(n_components=2)
    #data = np.matrix(data).T
    principalComp = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principalComp, columns=['PC1', 'PC2'])
    pca_df["label"] = labels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    unique_list = []
    for x in labels:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    colors = ['r', 'g', 'b']
    for target, color in zip(unique_list, colors):
        indicesToKeep = pca_df['label'] == target
        ax.scatter(pca_df.loc[indicesToKeep, 'PC1']
                   , pca_df.loc[indicesToKeep, 'PC2']
                   , c=color
                   , s=50)
    ax.legend(unique_list)
    ax.set_title(plot_title)
    ax.grid()
    plt.show()




filelist = []
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    if ".txt" in str(f):
        if not "result" in str(f) and not "new_dataset" in str(f):
            filelist.append(f);

print(f'{len(filelist)} files detected for clustering')
print(filelist)

epsilon = float(input('Enter the Cluster radius:'))
mpts = int(input('Enter number of minimum points: '))

for file in filelist:
    print("Performing DBSCAN on file " + file)
    nf_name = file.replace(".txt", "")
    #     print(nf_name)
    data = np.loadtxt(file, dtype='float')
    #     print(data)
    feature_matrix = data[:, 2:]
    cluster_id_list = dbscan(feature_matrix, epsilon, mpts, nf_name)
    #print(len(cluster_id_list))
    if("DBSCAN" in nf_name):
        plot_PCA_db(file, cluster_id_list, "Scatter Plot for DBSCAN on " + nf_name)
    else:
        plot_PCA(file, cluster_id_list, "Scatter Plot for DBSCAN on " + nf_name)
    #print(cluster_id_list)
    ground_truth = [item[1] for item in data]
    compute_similarity_coeff(cluster_id_list, ground_truth, nf_name)
