from Models import *
from Utils import *
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss, accuracy_score


def preProcessDataForLoop(train, test, valid):
    """
    split the data from the labels for train, test, validation
    and create the C vector with the theoretical C
    input:
        1. nd.array of train data concatenated with labels
        2. nd.array of test data concatenated with labels
        3. nd.array of valid data concatenated with labels
    output:
        1. nd.array of train data
        2. nd.array of train labels
        3. nd.array of test data
        4. nd.array of test labels
        5. nd.array of validation data
        6. nd.array of validation labels
        7. nd.array of the optional C values
        8. float representing the theoretical C value
    """
    C = np.linspace(0, 3, 60)
    theortical_C = getTheorticalLambda(train)
    C = np.append(C, theortical_C)
    train, train_lbl = train[:, :-1], train[:, -1]
    test, test_lbl = test[:, :-1], test[:, -1]
    val, val_lbl = valid[:, :-1], valid[:, -1]
    return train, train_lbl, test, test_lbl, val, val_lbl, C, theortical_C

np.random.seed(2020)

var_dict = {'mean': [np.array([0.3, 0.3]), np.array([2, 2])],
            'cov': [np.array([[0.1, 0.0], [0.0, 0.6]]), np.array([[0.5, 0.0], [0.0, 0.2]])]}
d_1, d_2 = multivariateNormalCreate(var_dict['mean'], var_dict['cov'], 5000)
multi_data = np.append(d_1, d_2, axis=0)
plotMultivariateData(multi_data, True)

data_path = os.path.join(os.getcwd(), "mnist_data")
desired_digits = [3, 8]
mnist_data = mnistPipeline(data_path, desired_digits)
desired_iris = [0, 1]
iris_data = irisPipeline(desired_iris)
desired_wine = [0, 2]
wine_data = winePipeline(desired_wine)

# split to train and test
split_factor = 0.8

np.random.shuffle(multi_data)
np.random.shuffle(mnist_data)
np.random.shuffle(iris_data)
np.random.shuffle(wine_data)
multi_train, multi_test, multi_val = trainTestValidationSplit(multi_data)
mnist_train, mnist_test, mnist_val = trainTestValidationSplit(mnist_data)
iris_train, iris_test, iris_val = trainTestValidationSplit(iris_data)
wine_train, wine_test, wine_val = trainTestValidationSplit(wine_data)

# experiment
for i in range(4):
    acc_test = np.array([])
    acc_val = np.array([])
    hinge = np.array([])
    if not i:
        # Multivariate
        train, train_lbl, test, test_lbl, val, val_lbl, C, C_theory = preProcessDataForLoop(multi_train,
                                                                                            multi_test,
                                                                                            multi_val)
    elif i == 1:
        # iris
        train, train_lbl, test, test_lbl, val, val_lbl, C, C_theory = preProcessDataForLoop(iris_train,
                                                                                            iris_test,
                                                                                            iris_val)
    elif i == 2:
        # wine
        train, train_lbl, test, test_lbl, val, val_lbl, C, C_theory = preProcessDataForLoop(wine_train,
                                                                                            wine_test,
                                                                                            wine_val)
    else:
        # MNIST
        train, train_lbl, test, test_lbl, val, val_lbl, C, C_theory = preProcessDataForLoop(mnist_train,
                                                                                            mnist_test,
                                                                                            mnist_val)

    C = C[1:]
    np.sort(C)
    C = C[np.where(C <= 1)]
    for c in C:
        svm_mdl = SVC(kernel='linear', C=c)
        svm_mdl.fit(train, train_lbl)
        hinge = np.append(hinge, hinge_loss(train_lbl, svm_mdl.predict(train)))
        acc_test = np.append(acc_test, accuracy_score(test_lbl, svm_mdl.predict(test)))
        acc_val = np.append(acc_val, accuracy_score(val_lbl, svm_mdl.predict(val)))

    if not i:
        exportPlots(C, C_theory, hinge, acc_test, acc_val, "Multivariate")
    elif i == 1:
        exportPlots(C, C_theory, hinge, acc_test, acc_val, "IRIS")
    elif i == 2:
        exportPlots(C, C_theory, hinge, acc_test, acc_val, "WINE")
    else:
        exportPlots(C, C_theory, hinge, acc_test, acc_val, "MNIST", True, list(map(str, desired_digits)))


print("Done")
