#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>

/**
 * Calculates Euclidean distance between two vectors.
 */
static double euclidean_distance(double *p, double *q, int d) {
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
        sum += pow(p[i] - q[i], 2);
    }
    return sqrt(sum);
}

/**
 * Python wrapper for the K-means algorithm fit method.
 * Expected arguments: (initial_centroids, data_points, K, iter, eps)
 */
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *py_centroids, *py_data;
    int K, iter_val, N, d;
    double eps;

    if (!PyArg_ParseTuple(args, "OOiid", &py_centroids, &py_data, &K, &iter_val, &eps)) {
        return NULL;
    }

    N = PyList_Size(py_data);
    d = PyList_Size(PyList_GetItem(py_data, 0));

    // Allocate memory for data and centroids
    double **data = (double **)malloc(N * sizeof(double *));
    double **centroids = (double **)malloc(K * sizeof(double *));
    double **new_centroids = (double **)malloc(K * sizeof(double *));
    int *cluster_counts = (int *)malloc(K * sizeof(int));

    if (!data || !centroids || !new_centroids || !cluster_counts) {
        return PyErr_NoMemory();
    }

    for (int i = 0; i < N; i++) {
        data[i] = (double *)malloc(d * sizeof(double));
        PyObject *row = PyList_GetItem(py_data, i);
        for (int j = 0; j < d; j++) {
            data[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    for (int i = 0; i < K; i++) {
        centroids[i] = (double *)malloc(d * sizeof(double));
        new_centroids[i] = (double *)malloc(d * sizeof(double));
        PyObject *row = PyList_GetItem(py_centroids, i);
        for (int j = 0; j < d; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    // Main K-means loop
    int iter_count = 0;
    int converged = 0;

    while (iter_count < iter_val && !converged) {
        // Reset clusters
        for (int i = 0; i < K; i++) {
            cluster_counts[i] = 0;
            for (int j = 0; j < d; j++) {
                new_centroids[i][j] = 0.0;
            }
        }

        // Assign to closest cluster
        for (int i = 0; i < N; i++) {
            int best_cluster = 0;
            double min_dist = euclidean_distance(data[i], centroids[0], d);
            
            for (int k = 1; k < K; k++) {
                double dist = euclidean_distance(data[i], centroids[k], d);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            cluster_counts[best_cluster]++;
            for (int j = 0; j < d; j++) {
                new_centroids[best_cluster][j] += data[i][j];
            }
        }

        // Update centroids and check convergence
        converged = 1;
        for (int i = 0; i < K; i++) {
            if (cluster_counts[i] > 0) {
                for (int j = 0; j < d; j++) {
                    new_centroids[i][j] /= cluster_counts[i];
                }
                
                if (euclidean_distance(centroids[i], new_centroids[i], d) >= eps) {
                    converged = 0;
                }
                
                for (int j = 0; j < d; j++) {
                    centroids[i][j] = new_centroids[i][j];
                }
            }
        }
        iter_count++;
    }

    // Build return Python list
    PyObject *result = PyList_New(K);
    for (int i = 0; i < K; i++) {
        PyObject *row = PyList_New(d);
        for (int j = 0; j < d; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(centroids[i][j]));
        }
        PyList_SetItem(result, i, row);
    }

    // Free memory
    for (int i = 0; i < N; i++) free(data[i]);
    for (int i = 0; i < K; i++) {
        free(centroids[i]);
        free(new_centroids[i]);
    }
    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_counts);

    return result;
}

static PyMethodDef MyKmeansMethods[] = {
    {"fit", (PyCFunction) fit, METH_VARARGS, 
     "Run k-means algorithm. Args: (initial_centroids, data_points, K, iter, eps)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mykmeanssp_module = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "K-means clustering C extension",
    -1,
    MyKmeansMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeanssp_module);
}
