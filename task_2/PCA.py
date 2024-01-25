import numpy as np

mean_vector = None
std_vector = None

# Step 1: Standardize the dataset
def standardize_data(X):
    global mean_vector
    global std_vector

    # Calculate mean and standard deviation along each feature (column)
    mean_vector = np.mean(X, axis=0)
    std_vector = np.std(X, axis=0)

    # Standardize the data
    X_std = (X - mean_vector) / std_vector

    return X_std

# Step 2: Compute the covariance matrix
def compute_covariance_matrix(X_std):
    return np.cov(X_std, rowvar=False, bias=True)

# Step 3: Compute eigenvalues and eigenvectors
def compute_eigen(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_index], eigenvectors[:, sorted_index]

# Step 4: Select principal components
def select_components(eigenvectors, num_components):
    return eigenvectors[:, :num_components]

# Step 5: Project data onto lower-dimensional linear subspace
def project_data(X_std, eigenvectors_selected):
    return np.dot(X_std, eigenvectors_selected)

#Combining all 5 steps. Returns the dataset and 3 vectors/matrice 
def pca (X, num_components):
    global mean_vector
    global std_vector

    # Step 1: Standardize the dataset
    X_std = standardize_data(X)

    # Step 2: Compute the covariance matrix
    cov_matrix = compute_covariance_matrix(X_std)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_eigen(cov_matrix)

    # Step 4: Select principal components
    eigenvectors_selected = select_components(eigenvectors, num_components)

    # Step 5: Project data onto lower-dimensional linear subspace
    X_pca = project_data(X_std, eigenvectors_selected)

    return X_pca, mean_vector, std_vector, eigenvectors_selected

#Reads dataset to an array
def file2arr(filename, count):
    f=open(filename, "r")
    s=f.readline()
    arr = []
    cls = []
    while s!="" and count>0:
        count -= 1
        s=s.split(",")
        for i in range(len(s)):
            s[i]=int(s[i])
        arr += [s[:-1]]
        cls += [s[-1]]
        s=f.readline()
    f.close()
    return arr, cls

#Saves the std vec, mean vec and matrice of eigen vectors to files
#all files have the same name (filename). the suffixes are: mean, std, eigvec 
def arr2files(filename, arr, cls, dim):
    b, mean, std, eigenvectors = pca(np.array(arr), dim)
    f=open(filename + ".ds" , "w")
    for row in range(len(b)):
        s = ""
        for col in range(len(b[row])):
            s += str(b[row][col]) + ","
        f.write(s + str(cls[row]) + "\n")
    f.close()

    f=open(filename + ".eigvec" , "w")
    for row in range(len(eigenvectors)):
        s = ""
        for col in range(len(eigenvectors[row])):
            s += str(eigenvectors[row][col]) + ","
        f.write(s[:-1] + "\n")
    f.close()
    
    f=open(filename + ".mean" , "w")
    s = ""
    for i in range(len(mean)):
        s += str(mean[i]) + ","
    f.write(s[:-1] + "\n")
    f.close()

    f=open(filename + ".std" , "w")
    s = ""
    for i in range(len(std)):
        s += str(std[i]) + ","
    f.write(s[:-1] + "\n")
    f.close()

#Reads datasetfrom file, converts using PCA and saves to files
def pcaDS(ds, length, dspcs, att):
    a, cls = file2arr(ds, length)
    arr2files(dspcs, a, cls, att)

#Reads std file
def read_std(filename):
    f=open(filename + ".std" , "r")
    s = f.readline().split(",")
    std = []
    for i in s:
        std += [float(i)]
    f.close
    return np.array(std)

#Reads mean file
def read_mean(filename):
    f=open(filename + ".mean" , "r")
    s = f.readline().split(",")
    mean = []
    for i in s:
        mean += [float(i)]
    f.close
    return np.array(mean)

#Reads eigvec file
def read_eigvec(filename):
    f=open(filename + ".eigvec" , "r")
    s = f.readline()
    eigvec  = []
    while s != "":
        s = s.split(",")
        vec = []
        for i in s:
            vec += [float(i)]
        eigvec += [vec]
        s = f.readline()
    f.close
    return np.array(eigvec)

#Converts an instance using PCA
def pca_instance(instance, filename):
    std = read_std(filename)
    mean = read_mean(filename)
    eigvec = read_eigvec(filename)
    return (np.dot( (np.array(instance) - mean) / std , eigvec )).tolist()

