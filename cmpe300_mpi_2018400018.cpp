/*
 * Student Name: Ege Can Kaya
 * Student Number: 2018400018
 * Compile Status: Compiling
 * Program Status: Working
 * Notes: N/A
 */

#include <stdio.h>
#include "mpi.h"
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

// given a long double vector, finds the T greatest elements and returns their indexes
void find_T_max(std::vector<int> &indexes, std::vector<long double> weights, int T) {
    for (int t = 0; t < T; t++) {
        long double max = std::numeric_limits<long double>::min();
        int max_index = -1;
        for (int i = 0; i < weights.size(); i++) {
            if (weights.at(i) > max) {
                max = weights.at(i);
                max_index = i;
            }
        }
        indexes.push_back(max_index);
        weights.at(max_index) = std::numeric_limits<long double>::min();
    }
}

// the diff helper function described in the assignment description
long double diff(int a, std::vector<long double> i1, std::vector<long double> i2, long double min, long double Max) {
    return std::fabs(i1.at(a) - i2.at(a)) / (Max - min); // don't forget to use std::fabs(), and not ::abs()!
}

// given two long double vectors, returns their l1 (Manhattan) distance (disregarding the last element)
long double l1_dist(std::vector<long double> v1, std::vector<long double> v2) {
    long double d = 0.0;
    for (int i = 0; i < v1.size() - 1; i++) {
        d += std::fabs(v1.at(i) - v2.at(i));
    }
    return d;
}

// given a matrix of instances, returns the minimum and maximum values of the attribute a in all instances
void min_and_max(std::pair<long double, long double> &mM, int a, std::vector <std::vector<long double>> instances) {
    long double max = std::numeric_limits<long double>::min();
    long double min = std::numeric_limits<long double>::max();
    for (int i = 0; i < instances.size(); i++) {
        long double attribute = instances.at(i).at(a);
        if (attribute > max) max = attribute;
        if (attribute < min) min = attribute;
    }

    mM = {min, max};
}

// the relief algorithm described in the assignment description
void relief(std::vector<long double> &weights, std::vector <std::vector<long double>> instances, int A, int M) {
    std::fill(weights.begin(), weights.end(), 0); // set all weights to 0
    for (int m = 0; m < M; m++) {
        std::vector<long double> target_instance = instances.at(m);
        // find nearest hit and miss by traversing all instances in this processor
        std::vector<long double> hit, miss;
        long double h_distance = std::numeric_limits<long double>::max();
        long double m_distance = std::numeric_limits<long double>::max();

        for (int i = 0; i < instances.size(); i++) {
            if (i == m) continue; // no need to calculate distance from itself
            else {
                std::vector<long double> compared_instance = instances.at(i);
                long double distance = l1_dist(target_instance, compared_instance);
                // if we find a new nearest with the same class value, it's a hit
                if (distance <= h_distance && (int) target_instance.at(A) == (int) compared_instance.at(A)) {
                    h_distance = distance;
                    hit = compared_instance;
                    // if we find a new nearest with a different class value, it's a miss
                } else if (distance <= m_distance && (int) target_instance.at(A) != (int) compared_instance.at(A)) {
                    m_distance = distance;
                    miss = compared_instance;
                }

            }
        }

        // update all the attribute weights with regard to this instance
        for (int a = 0; a < A; a++) {
            std::pair<long double, long double> mM;
            min_and_max(mM, a, instances);
            long double min = mM.first;
            long double Max = mM.second;

            weights.at(a) = weights.at(a)
                            - (diff(a, target_instance, hit, min, Max) / M)
                            + (diff(a, target_instance, miss, min, Max) / M);
        }

    }
}

int main(int argc, char *argv[]) {
    std::vector <std::vector<long double>> instances; // instances is a matrix of features vectors
    std::vector <std::vector<long double>> partition; // this will hold the partitions for each slave process
    int A, P, N, M, T;
    int partition_size;
    int rank; // rank of the current processor
    int size; // total number of processors

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets the rank of the current processor
    MPI_Comm_size(MPI_COMM_WORLD, &size); // gets the total number of processors

    // ****************************************** //

    // if it's master processor, read from input file
    if (rank == 0) {
        std::string input_path = argv[1];
        std::ifstream input_file(input_path);
        if (input_file.is_open()) {
            std::string N_str, A_str, M_str, T_str, line, s;
            getline(input_file, line);
            P = stoi(line);
            getline(input_file, line);
            std::stringstream split_line(line);
            split_line >> N_str >> A_str >> M_str >> T_str;
            N = stoi(N_str);
            A = stoi(A_str);
            M = stoi(M_str);
            T = stoi(T_str);
            partition_size = N / (P - 1); // calculate the number of lines that will be held in each slave

            // read the data for the following N instances and put them in the instances matrix
            for (int i = 0; i < N; i++) {
                getline(input_file, line);
                std::vector<long double> instance;
                std::stringstream split_line(line);

                for (int j = 0; j < A + 1; j++) {
                    split_line >> s;
                    instance.push_back(stold(s));

                }

                instances.push_back(instance);
            }
        }

        // send the slave processors their partition
        for (int i = 0; i < N; i++) {
            int target_processor = (i / partition_size) + 1;
            MPI_Send(instances.at(i).data(), A + 1, MPI_LONG_DOUBLE, target_processor, 0, MPI_COMM_WORLD);
        }

    }

    // broadcast the values of P, A, M, N, T and partition_size
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&partition_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // A attributes, each starting with value 0
    std::vector<long double> weights(A, 0);

    // all slave processes will execute this block
    if (rank != 0) {

        // create an instance vector that will hold A+1 long doubles (features)
        std::vector<long double> instance;
        instance.resize(A + 1);

        // populate the partitions
        for (int i = 0; i < partition_size; i++) {
            MPI_Recv(&instance[0], A + 1, MPI_LONG_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partition.push_back(instance);
        }

        // each process runs the relief algorithm concurrently on their partition
        relief(weights, partition, A, M);

        // get the indexes of the greatest T elements in the weights vector and sort them
        std::vector<int> indexes;
        find_T_max(indexes, weights, T);
        std::sort(indexes.begin(), indexes.end());

        std::cout << "Slave P" << rank << " : ";
        for (int i = 0; i < indexes.size(); i++) {
            std::cout << indexes.at(i);
            if (i != indexes.size() - 1) std::cout << " ";
        }
        std::cout << std::endl;

        // send the results back to the master
        MPI_Send(indexes.data(), T, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // back to the master process
    if (rank == 0) {

        // receive the indexes of the T highest weights from all slaves
        std::vector<int> result;
        result.resize(T * (P - 1));
        for (int i = 0; i < P - 1; i++) {
            MPI_Recv(&result[i * T], T, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // sort the indexes and remove duplicates
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());

        std::cout << "Master P0" << " : ";
        for (int i = 0; i < result.size(); i++) {
            std::cout << result.at(i);
            if (i != result.size() - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    // ****************************************** //

    MPI_Barrier(MPI_COMM_WORLD); // synchronizing processes
    MPI_Finalize();

    return 0;
}
