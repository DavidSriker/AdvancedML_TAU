from Models import *

X = np.array([[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
gt = np.array([-1.0, -1.0, -1.0, 1.0])

# test preceptron
print(20 * '-', 'Perceptron', 20 * '-')
perceptron_mdl = Perceptron(lr=2, num_epochs=8)
perceptron_mdl.fit(X, gt)
print("results:")
for i, sample in enumerate(X):
    print(perceptron_mdl.predict(sample), gt[i])

print(20 * '-', 'SVM', 20 * '-')
# test binary svm
svm_mdl = SVM(C=2)
svm_mdl.fit(X, gt)
print("results:")
for i, pred in enumerate(svm_mdl.predict(X)):
    print(pred, gt[i])
