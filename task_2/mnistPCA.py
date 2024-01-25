import PCA
import knn


#using PCA on mnisit dataset

for dim in range(5,35,5):

    # Apply PCA
    pca = PCA(n_components=dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)


    maxAcc=0
    maxK=0

    for k in range(5,9):

        # Train k-NN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = knn_classifier.predict(X_test_pca)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Update max accuracy and corresponding k
        if accuracy > maxAcc:
            maxAcc = accuracy
            maxK = k
        print("dim =", dim, " The best k is:", maxK, " The accuracy is:", maxAcc)

        # Run k-NN on the first 30 lines in testing.txt 6 times
        for j in range(6):
            # Load the first 30 lines of testing.txt
            testing_data = np.loadtxt('testing.txt', delimiter=',', max_rows=30)

            # Separate features and labels
            X_test_custom = testing_data[:, :-1]
            y_test_custom = testing_data[:, -1]

            # Apply PCA to custom test data
            X_test_custom_pca = pca.transform(X_test_custom)

            # Train k-NN classifier
            knn_classifier_custom = KNeighborsClassifier(n_neighbors=maxK)
            knn_classifier_custom.fit(X_train_pca, y_train)

            # Make predictions on custom test set
            y_pred_custom = knn_classifier_custom.predict(X_test_custom_pca)

            # Evaluate accuracy on custom test set
            accuracy_custom = accuracy_score(y_test_custom, y_pred_custom)

            print("Run", j + 1, "Accuracy on custom test set:", accuracy_custom)


    for j in range(30):
        # Make predictions on custom test set
        y_pred_custom = knn_classifier_custom.predict(X_test_custom_pca)

        # Evaluate accuracy on custom test set
        accuracy_custom = accuracy_score(y_test_custom, y_pred_custom)

        print("\nAccuracy on custom test set:", accuracy_custom)
        print("True labels:", y_test_custom.astype(int))
        print("Predicted labels:", y_pred_custom.astype(int))




    
    print("dim= ",dim," The best k is: ",maxK," The accuracy is: ",maxAcc)
    

