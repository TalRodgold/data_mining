# Binyamin Mor - 317510485
# Tal Rodgold - 318162344

import PCA
import knn

# using PCA on mnisit dataset

# Read MNIST dataset
mnist_data, mnist_labels = PCA.file2arr('digits-testing.txt', 30)

for dim in range(5, 35, 5):

    print("dim = ", dim, "\n")
    # Apply PCA
    PCA.pcaDS('digits-training.txt', 10000, 'training_pca', dim)

    maxAcc = 0
    maxK = 0

    # We will check by 5 nearest neighbors until 8 nearest neighbors
    for k in range(5, 9):
        print("-------")
        print("k = ", k, "\n")

        acc = 0
        # Go over all the rows
        for j in range(30):

            data_testing_pca_line = PCA.pca_instance(mnist_data[j], 'training_pca')
            #Apply KNN
            result = knn.knn(k, 'training_pca.ds', data_testing_pca_line)

            print("digit: ", mnist_labels[j], ", knn= ", result)

            if (result == mnist_labels[j]):
                acc += 1

        accuracy = acc / 30
        print("\naccuracy = ", accuracy, "\n")

        # Find the most accurate K
        if (accuracy > maxAcc):
            maxAcc = accuracy
            maxK = k

    print("dim = ", dim, " The best k is: ", maxK, " The accuracy is: ", maxAcc)
    print("*********************************\n")

