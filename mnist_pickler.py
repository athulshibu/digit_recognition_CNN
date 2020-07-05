import pickle
import numpy
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "C:/Users/HP/Desktop/Athul/Academics B.Tech/Semester 6/Introduction to Data Communication/Mini Project - Digit Recognition/"

train_data = numpy.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = numpy.loadtxt(data_path + "mnist_test.csv", delimiter=",") 
factor = 0.99 / 255
train_imgs = numpy.asfarray(train_data[:, 1:]) * factor + 0.01
test_imgs = numpy.asfarray(test_data[:, 1:]) * factor + 0.01

train_labels = numpy.asfarray(train_data[:, :1])
test_labels = numpy.asfarray(test_data[:, :1])
lr = numpy.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(numpy.float)
test_labels_one_hot = (lr==test_labels).astype(numpy.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99



with open(data_path + "pickled_mnist.pkl", "bw") as line:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels,
            train_labels_one_hot,
            test_labels_one_hot)
    pickle.dump(data, line)


print("Pickled!!!!")