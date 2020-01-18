from Models import *
from Utils import *
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import hinge_loss, accuracy_score
from sklearn.model_selection import GridSearchCV

np.random.seed(2020)

var_dict = {'mean': [np.array([0.3, 0.3]), np.array([2, 2])],
            'cov': [np.array([[0.1, 0.0], [0.0, 0.6]]), np.array([[0.5, 0.0], [0.0, 0.2]])]}
d_1, d_2 = multivariateNormalCreate(var_dict['mean'], var_dict['cov'], 5000)
multi_data = np.append(d_1, d_2, axis=0)
plotMultivariateData(multi_data, True)

data_path = os.path.join(os.getcwd(), "mnist_data")
desired_digits = [2, 9]
mnist_data = mnistPipeline(data_path, desired_digits)

# split to train and test
split_factor = 0.8

np.random.shuffle(multi_data)
np.random.shuffle(mnist_data)
multi_train, multi_test, multi_val = trainTestValidationSplit(multi_data)
mnist_train, mnist_test, mnist_val = trainTestValidationSplit(mnist_data)

# experinment
for i in range(2):
    acc_test = np.array([])
    acc_val = np.array([])
    hinge = np.array([])
    if not i:
        # Multivariate
        C = np.linspace(0,3,60)
        theortical_C = getTheorticalLambda(multi_train)
        C = np.append(C, theortical_C)
        train, train_lbl = multi_train[:, :-1], multi_train[:, -1]
        test, test_lbl = multi_test[:, :-1], multi_test[:, -1]
        val, val_lbl = multi_val[:, :-1], multi_val[:, -1]
    else:
        # MNIST
        C = np.linspace(0,5,60)
        theortical_C = getTheorticalLambda(mnist_train)
        C = np.append(C, theortical_C)
        train, train_lbl = mnist_train[:, :-1], mnist_train[:, -1]
        test, test_lbl = mnist_test[:, :-1], mnist_test[:, -1]
        val, val_lbl = mnist_val[:, :-1], mnist_val[:, -1]

    C = C[1:]
    np.sort(C)
    for c in C:
        svm_mdl = SVC(kernel='linear', C=c)
        svm_mdl.fit(train, train_lbl)
        hinge = np.append(hinge, hinge_loss(train_lbl, svm_mdl.predict(train)))
        acc_test = np.append(acc_test, accuracy_score(test_lbl, svm_mdl.predict(test)))
        acc_val = np.append(acc_val, accuracy_score(val_lbl, svm_mdl.predict(val)))

    exportPlots(C, theortical_C, hinge, acc_test, acc_val, "Multivariate" if not i else "MNIST")
print("Done")
