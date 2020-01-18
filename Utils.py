import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import os


def normalizeByMax(data):
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
    mapping= {-1.0: ("red", "x"), 1.0: ("blue", "o")}
    for l in mapping.keys():
        x = data[data[:, 2] == l]
        plt.scatter(x[:,0], x[:,1], c=mapping[l][0], marker=mapping[l][1])
    if not save_flag:
        plt.show()
        time.sleep(5.5)
    else:
        im_path = os.path.join(os.getcwd(), "images")
        if not os.path.exists(im_path):
            os.makedirs(im_path)
        plt.savefig(os.path.join(im_path, 'MultivariateData.png'))
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
    assert len(digits)==2, "should get only 2 digits"
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
        1. a nd.array of data concatenated with labels
    """
    data_path = os.path.join(os.getcwd(), "mnist_data")
    train_im, train_lbl, test_im, test_lbl = readMnistData(dir_path)
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
    num_sampeles = data.shape[0]
    train = data[:int(num_sampeles * train_frac)]
    test = data[int(num_sampeles * train_frac):]
    num_samples = train.shape[0]
    validation = train[:int(num_sampeles * validation_frac)]
    train = train[int(num_sampeles * validation_frac):]
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

def exportPlots(C, C_theory, loss, test_acc, val_acc, data_name):
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
    print("Best Accuaracy on test: {:} for C: {:}".format(test_acc[np.argmax(test_acc)],
                                                            C[np.argmax(test_acc)]))
    print("Best Accuaracy on validation: {:} for C: {:}".format(val_acc[np.argmax(val_acc)],
                                                                C[np.argmax(val_acc)]))

    plt.scatter(C, val_acc, color='k')
    plt.scatter(C, test_acc, color='r')
    plt.savefig(os.path.join(im_path, '{:}_acc.png'.format(data_name)))
    plt.close()

    plt.scatter(C, loss, color='k')
    plt.savefig(os.path.join(im_path, '{:}_loss.png'.format(data_name)))
    plt.close()
    return
