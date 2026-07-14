import sys
import numpy as np
import pandas as pd
import mykmeanssp

def exit_with_error(msg="An Error Has Occurred"):
    """Prints error message and terminates the program."""
    print(msg)
    sys.exit(1)

def main():
    # 1. Reading user CMD arguments
    args = sys.argv[1:]
    if len(args) == 4:
        # K is not provided, default is 3
        K = 3
        iter_idx, eps_idx, f1_idx, f2_idx = 0, 1, 2, 3
    elif len(args) == 5:
        try:
            K = int(args[0])
        except ValueError:
            exit_with_error("Invalid number of clusters!")
        iter_idx, eps_idx, f1_idx, f2_idx = 1, 2, 3, 4
    else:
        exit_with_error()

    try:
        iter_val = int(args[iter_idx])
        eps = float(args[eps_idx])
        file_name_1 = args[f1_idx]
        file_name_2 = args[f2_idx]
    except ValueError:
        exit_with_error()

    # Validations
    if iter_val <= 1 or iter_val >= 400:
        exit_with_error("Incorrect maximum iteration!")
    if eps < 0:
        exit_with_error("Incorrect epsilon!")

    # 2. & 3. Combine both input files and sort
    try:
        df1 = pd.read_csv(file_name_1, header=None)
        df2 = pd.read_csv(file_name_2, header=None)
        
        # Inner join using the first column as key
        df = pd.merge(df1, df2, on=0, how='inner')
        # Sort by the 'key' (first column) in ascending order
        df = df.sort_values(by=0).reset_index(drop=True)
    except Exception:
        exit_with_error()

    keys = df[0].values.astype(int)
    data = df.drop(columns=[0]).values.astype(float)
    N = data.shape[0]

    if K <= 1 or K >= N:
        exit_with_error("Incorrect number of clusters!")

    # 4. K-means++ implementation
    np.random.seed(1234)
    
    # (a) Choose one center uniformly at random
    initial_indices = [np.random.choice(N)]
    centroids = [data[initial_indices[0]]]

    for _ in range(1, K):
        # (b) Compute minimal distance D(x) to already chosen centers
        # d(p,q) = sqrt(...) based on assignment
        distances = np.linalg.norm(data - centroids[0], axis=1)
        for j in range(1, len(centroids)):
            distances = np.minimum(distances, np.linalg.norm(data - centroids[j], axis=1))
        
        # (c) Choose new center with weighted probability
        probabilities = distances / np.sum(distances)
        chosen_idx = np.random.choice(N, p=probabilities)
        
        initial_indices.append(chosen_idx)
        centroids.append(data[chosen_idx])

    # Convert to Python lists to pass to C extension
    centroids_list = [c.tolist() for c in centroids]
    data_list = data.tolist()

    # 5. Interface with C extension
    try:
        final_centroids = mykmeanssp.fit(centroids_list, data_list, K, iter_val, eps)
    except Exception:
        exit_with_error()

    # 6. Outputting the results
    # First line: indices of the observations chosen as initial centroids
    print(",".join(map(str, [keys[i] for i in initial_indices])))
    
    # Second line onwards: calculated final centroids
    for centroid in final_centroids:
        print(",".join(["%.4f" % val for val in centroid]))

if __name__ == "__main__":
    main()
