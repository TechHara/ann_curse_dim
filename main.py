import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.sum(x**2, axis=-1, keepdims=True)**0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=str, help='comma-separated dims', default='2,3,4,5,10,20,50,100,200,500')
    parser.add_argument('--clusters', type=str, help='comma-separated # clusters', default='10,20,50,100')
    parser.add_argument('--ndb', type=int, default=100000, help='# database')
    parser.add_argument('--nq', type=int, default=1000, help='# queries')
    parser.add_argument('--num_train', type=int, default=0, help='# data used for clustering')
    parser.add_argument('--normalize', action='store_true', help='normalize db vectors')

    args = parser.parse_args()
    dims = sorted([int(n) for n in args.dims.split(',')])
    clusters = sorted([int(n) for n in args.clusters.split(',')])
    if args.num_train == 0:
        args.num_train = np.max(clusters) * 100
    
    recall = np.zeros((len(dims), len(clusters)), dtype=np.float32)
    for i,dim in enumerate(dims):
        for j,nc in enumerate(clusters):
            recall[i,j] = run(dim, nc, args)

    np.savetxt('/dev/stdout', recall, delimiter='\t')


def run(dim: int, nc: int, args):
    db = np.random.rand(args.ndb, dim).astype(np.float32) - 0.5
    if args.normalize: db = l2_normalize(db)
    queries = np.random.rand(args.nq, dim).astype(np.float32) - 0.5

    index = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='l2').fit(db)
    _, nn_indices = index.kneighbors(queries)
    nn_indices = nn_indices.flatten()

    kmeans = KMeans(nc, n_init='auto').fit(db[:args.num_train])
    db_cluster_indices = kmeans.predict(db)[nn_indices]
    q_cluster_indices = kmeans.predict(queries)

    return np.mean(db_cluster_indices == q_cluster_indices)


if __name__ == '__main__':
    main()