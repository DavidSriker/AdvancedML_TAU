import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist, iris_data
import os
import time


def normalizeByMax(data):
    """
    normalize the data matrix row-wise
    input:
        1. nd.array of data
    output:
        1. nd.array of normalized data
    """
    return data / np.max(data)

# multivariate normal distribution
def multivariateNormalCreate(means, covariances, num_samples):
    """
    create 2 multivariate normal distribution data
    input:
        1. dict of means values
        2. dict of covariance matrices
    output:
        1. return 2 nd.array of the generated data with size (num_samples, 3) each
        the third column is the label
    """
    if num_samples % 2:
        num_x_1 = round(num_samples / 2)
        num_x_2 = num_samples - num_x_1
    else:
        num_x_1 = num_x_2 = round (num_samples / 2)

    x_1 = normalizeByMax(np.random.multivariate_normal(means[0],
                                                        covariances[0],
                                                        num_x_1))
    x_2 = normalizeByMax(np.random.multivariate_normal(means[1],
                                                        covariances[1],
                                                        num_x_2))
    return setLabels(x_1, 1.0), setLabels(x_2, -1.0)

def setLabels(data, label):
    """
    give a label column to generated data
    input:
        1. nd.array of generated data
        2. desired label
    output:
        1. return the data with added column with the desired label
    """
    labels_vec = np.ones((data.shape[0], 1)) * label
    return np.append(data, labels_vec, axis=1)

def plotMultivariateData(data, save_flag=False):
    """
    plot the generated data in scatter plot with different color and marker
    for each label
    input:
        1. entire data with labels
        2. boolean indicating to save the image
    output:
        1. void
    """
    mapping = {-1.0: ("red", "x"), 1.0: ("blue", "o")}
    for l in mapping.keys():
        x = data[data[:, 2] == l]
        plt.scatter(x[:, 0], x[:, 1], c=mapping[l][0], marker=mapping[l][1], s=10)
        plt.grid()
        plt.xlabel("Position [Arb.]")
        plt.ylabel("Position [Arb.]")
        plt.legend(["Gauss 1", "Gauss 2"])
        plt.title("Non Linearly Separable Gaussian Distribution")
    if not save_flag:
        plt.show()
        time.sleep(5.5)
    else:
        im_path = os.path.join(os.getcwd(), "images")
        if not os.path.exists(im_path):
            os.makedirs(im_path)
        plt.savefig(os.path.join(im_path, 'MultivariateData.eps'), format='eps')
    plt.close()
    return

# MNIST
def readMnistData(dir_path):
    """
    reads the mnist data files
    input:
        1. directory contain all the mnist data files
    output:
        1. training images array
        2. training labels array
        3. test images array
        4. test labels array
    """
    files = os.listdir(dir_path)
    train_files = [f for f in files if "train" in f]
    test_files = [f for f in files if "t10k" in f]
    train_files.sort()
    test_files.sort()

    x_train, y_train = loadlocal_mnist(images_path=os.path.join(dir_path, train_files[0]),
                                        labels_path=os.path.join(dir_path, train_files[1]))
    x_test, y_test = loadlocal_mnist(images_path=os.path.join(dir_path, test_files[0]),
                                        labels_path=os.path.join(dir_path, test_files[1]))
    return x_train, y_train, x_test, y_test

def sortMnistData(images, labels):
    """
    sort the images and labels based on the labels values
    input:
        1. images data
        2. labels data
    output:
        1. sorted images data
        2. sorted labels data
    """
    sorted_idx = np.argsort(labels)
    return images[sorted_idx], labels[sorted_idx]

def findLabelSlices(labels):
    """
    find the index slices of each label
    input:
        1. entire labels data
    output:
        1. a list that each element is a list with the start and end position
        of the label
    """
    desired_idx = []
    for lbl in np.unique(labels):
        pos = np.where(labels == lbl)
        desired_idx.append([pos[0][0], pos[0][-1]])
    return desired_idx

def getDesiredDigitsData(images, digits, slices):
    """
    get the desired 2 digits data and set the labels to -1,1
    input:
        1. the images data
        2. the desired 2 digits
        3. the digits index slices
    output:
        1. a dictionary that contain the data for the 2 digits with the labels
        being -1 or 1
    """
    assert len(digits) == 2, "should get only 2 digits"
    data_dict = {}
    for i, d in enumerate(digits):
        sliced_im = images[np.arange(*slices[d])]
        binarized_lbl = np.ones((sliced_im.shape[0],)) * (-1 if i else 1)
        data_dict[i] = {"im": sliced_im, "lbl": binarized_lbl}
    return data_dict

def mnistPipeline(dir_path, desired_digits):
    """
    preprocess the data
    input:
        1. the mnist files directory
        2. the desired digits to extract
    output:
        1. nd.array of data concatenated with labels
    """
    data_path = os.path.join(os.getcwd(), "mnist_data")
    train_im, train_lbl, test_im, test_lbl = readMnistData(dir_path)
    generateMnistFigure(train_im, train_lbl)
    train_im = normalizeByMax(train_im)
    train_im, train_lbl = sortMnistData(train_im, train_lbl)
    # test_im, test_lbl = sortMnistData(test_im, test_lbl)
    train_slices = findLabelSlices(train_lbl)
    # test_slices = findLabelSlices(test_lbl)
    binary_data = getDesiredDigitsData(train_im, desired_digits, train_slices)

    data = np.concatenate((binary_data[0]["im"],
                            binary_data[1]["im"]))
    labels = np.concatenate((binary_data[0]["lbl"],
                            binary_data[1]["lbl"]))
    return np.append(data, labels.reshape((labels.shape[0], 1)), axis=1)

def irisPipeline(desired_lbl):
    """
    divide the data to train, test and validation sets
    input:
        1. a list of the desired iris lables: Setosa - 0
                                              Versicolor - 1
                                              Virginica - 2
    output:
        1. nd.array of data concatenated with labels
    """
    assert len(desired_lbl) == 2, "should get only 2 labels"
    data, lbl = iris_data()
    data = normalizeByMax(data)
    sliced_lbl = np.logical_or(lbl == desired_lbl[0], lbl == desired_lbl[1])
    sliced_data = data[sliced_lbl, :]
    sliced_lbl = lbl[sliced_lbl]
    return np.append(sliced_data, sliced_lbl.reshape(sliced_lbl.shape[0], 1), axis=1)

def trainTestValidationSplit(data, train_frac=0.8, validation_frac=0.1):
    """
    divide the data to train, test and validation sets
    input:
        1. data matrix
        2. training fraction for the splitting
        3. validation fraction for the splitting
    output:
        1. train matrix
        2. test matrix
        3. validation matrix
    """
    data_num_samples = data.shape[0]
    train = data[:int(data_num_samples * train_frac)]
    test = data[int(data_num_samples * train_frac):]
    train_num_samples = train.shape[0]
    validation = train[:int(train_num_samples * validation_frac)]
    train = train[int(train_num_samples * validation_frac):]
    return train, test, validation

def getTheorticalLambda(data):
    """
    find the best theortical lambda for the regulizer based on theory
    input:
        1. data matrix
    output:
        1. the theortical best lambda
    """
    B = np.max(np.linalg.norm(data[:, :-1], axis=1))
    L = 1
    n = data.shape[0]
    C = np.sqrt((8 * L * B ** 2) / n)
    return C

def exportPlots(C, C_theory, loss, test_acc, val_acc, data_name, mnist_flag=False, digits=[]):
    """
    export the experiment plots to images directory
    input:
        1. vector of the used C values
        2. theortical best C value
        3. loss values vector
        4. test accuracy values vector
        5. validation accuracy values vector
        6. string that indicate what is the source of the data
    output:
        void
    """
    im_path = os.path.join(os.getcwd(), "images")
    if not os.path.exists(im_path):
        os.makedirs(im_path)

    print(20 * "*", data_name, 20 * "*")
    print("Theoretical C: {:}".format(C_theory))
    print("Best Accuaracy on test: {:} for C: {:}, while validation is: {:}".format(test_acc[np.argmax(test_acc)],
                                                                                    C[np.argmax(test_acc)], val_acc[np.argmax(test_acc)]))
    print("Best Accuaracy on validation: {:} for C: {:}, while test is: {:}".format(val_acc[np.argmax(val_acc)],
                                                                                    C[np.argmax(val_acc)], test_acc[np.argmax(val_acc)]))
    print("Accuaracy on validation and test (theory): valid - {:}, test - {:}".format(val_acc[np.where(C_theory == C)],
                                                                             test_acc[np.where(C_theory == C)]))

    theory_idx = np.where(C == C_theory)
    top_val_idx = np.argmax(val_acc)
    top_test_idx = np.argmax(test_acc)
    for idx, c in enumerate(C):
        if idx not in [top_val_idx, top_test_idx, *theory_idx] and not np.isclose(c, C_theory, rtol=10E-2):
            plt.scatter(c, val_acc[idx], color='k', marker="o", s=20)
            plt.scatter(c, test_acc[idx], color='r', marker="o", s=20)
        elif idx == top_val_idx:
            plt.scatter(c, val_acc[idx], color='b', s=20, marker="d", label="Top Valid. Coeff (valid)")
            plt.scatter(c, test_acc[idx], color='c', s=20, marker="d", label="Top Valid. Coeff (test)")
        elif idx == top_test_idx:
            plt.scatter(c, val_acc[idx], color='g', s=20, marker="d", label="Top Valid. Coeff (valid)")
            plt.scatter(c, test_acc[idx], color='purple', s=20, marker="d", label="Top Valid. Coeff (test)")

    plt.scatter(c, val_acc[theory_idx[0]], color='m', s=20, marker="X", label="Theory Coeff (valid)")
    plt.scatter(c, test_acc[theory_idx[0]], color='y', s=20, marker="X", label="Theory Coeff (test)")

    plt.legend()
    plt.xlabel(r"$\lambda$ Value")
    plt.ylabel("Accuracy [%]")
    plt.grid()
    plt.title(r"$\lambda$ Value Vs. Validation Accuracy (Black) and Test Accuracy (Red)")
    if not mnist_flag:
        plt.savefig(os.path.join(im_path, '{:}_acc.eps'.format(data_name)), format='eps')
    else:
        plt.savefig(os.path.join(im_path, '{:}_acc_digits_{:}_{:}.eps'.format(data_name, *digits)), format='eps')
    plt.close()

    top_loss_idx = np.argmin(loss)
    for idx, c in enumerate(C):
        if idx not in [top_loss_idx, theory_idx[0]]:
            plt.scatter(c, loss[idx], color='k', marker="o", s=15)
        elif idx == top_loss_idx:
            plt.scatter(c, loss[top_loss_idx], color='g', s=15, marker="d", label="Min. Loss Real")

    plt.scatter(c, loss[theory_idx[0]], color='m', s=15, marker="X", label="Min. Loss Theory")

    plt.legend()
    plt.xlabel(r"$\lambda$ Value")
    plt.ylabel("Accuracy [%]")
    plt.grid()
    plt.title(r"$\lambda$ Value Vs. Loss")
    if not mnist_flag:
        plt.savefig(os.path.join(im_path, '{:}_loss.eps'.format(data_name)), format='eps')
    else:
        plt.savefig(os.path.join(im_path, '{:}_loss_digits_{:}_{:}.eps'.format(data_name, *digits)), format='eps')
    plt.close()

    return

def generateMnistFigure(mnist_data, mnist_labels):
    im_path = os.path.join(os.getcwd(), "images")
    if not os.path.exists(im_path):
        os.makedirs(im_path)

    fig = plt.figure(figsize=(7, 7))
    for i in range(0, 10):
        idx = np.where(mnist_labels == i)[0][np.random.randint(0, 100)]
        ax = fig.add_subplot(2, 5, i + 1)
        ax.set_aspect("equal")
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        plt.tick_params(
            axis='y',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        plt.axis('off')
        plt.imshow(mnist_data[idx].reshape(28, 28), cmap='plasma')
        plt.title("Class is: " + str(mnist_labels[idx]))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1, top=0.6, hspace=0)
    plt.savefig(os.path.join(im_path, 'MNIST.eps'), format='eps', bbox_inches='tight',
                pad_inches=0)
    plt.close(fig)
    return
