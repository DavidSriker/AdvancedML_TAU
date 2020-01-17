from Models import *
from Utils import *
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import hinge_loss, accuracy_score

var_dict = {'mean': [np.array([0.5, 0.5]), np.array([2, 2])],
            'cov': [np.array([[0.4, 0.0], [0.0, 0.6]]), np.array([[0.5, 0.0], [0.0, 0.2]])]}
d_1, d_2 = multivariateNormalCreate(var_dict['mean'], var_dict['cov'], 1000)
multi_data = np.append(d_1, d_2, axis=0)
plotMultivariateData(multi_data, True)

data_path = os.path.join(os.getcwd(), "mnist_data")
desired_digits = [2, 9]
mnist_data = mnistPipeline(data_path, desired_digits)


print(20 * '-', 'SVM Multivariate Normal Distribution', 20 * '-')
np.random.shuffle(multi_data)
svm_mdl_multi = LinearSVC(C=0.01)
svm_mdl_multi.fit(multi_data[:, :-1], multi_data[:, -1])
print("acc: ", accuracy_score(multi_data[:, -1], svm_mdl_multi.predict(multi_data[:, :-1])))
print("hinge loss:", hinge_loss(multi_data[:, -1], svm_mdl_multi.predict(multi_data[:, :-1])))


print(20 * '-', 'SVM MNIST', 20 * '-')
np.random.shuffle(mnist_data)
svm_mdl_mnist = SVC(C=0.01)
svm_mdl_mnist.fit(mnist_data[:, :-1], mnist_data[:, -1])
print("acc: ", accuracy_score(mnist_data[:, -1], svm_mdl_mnist.predict(mnist_data[:, :-1])))
print("hinge loss:", hinge_loss(mnist_data[:, -1], svm_mdl_mnist.predict(mnist_data[:, :-1])))
