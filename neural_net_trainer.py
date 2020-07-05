import numpy
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import pickle
from scipy.stats import truncnorm

#@np.vectorize
def sigmoid(x):
    return 1 / (1 + numpy.e ** -x)

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def create_weight_matrices(no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes):
    rad = 1 / numpy.sqrt(no_of_in_nodes)
    X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
    wih = X.rvs((no_of_hidden_nodes, no_of_in_nodes)) # Weight from Input to Hidden Layer
    
    rad = 1 / numpy.sqrt(no_of_hidden_nodes)
    X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
    who = X.rvs((no_of_out_nodes, no_of_hidden_nodes)) # Weight from Hidden to Output Layer
    return wih, who

def train(wih, who, learning_rate, input_vector, target_vector):
    input_vector = numpy.array(input_vector, ndmin=2).T # Input Vector needs to be Transposed
    target_vector = numpy.array(target_vector, ndmin=2).T # Target Vector needs to be Transposed
    
    output_hidden = sigmoid(numpy.dot(wih, input_vector))
    output_network = sigmoid(numpy.dot(who, output_hidden))
    
    output_errors = target_vector - output_network
    tmp = output_errors * output_network * (1.0 - output_network)     
    who += learning_rate  * numpy.dot(tmp, output_hidden.T)

    hidden_errors = numpy.dot(who.T, output_errors)
    tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
    wih += learning_rate * numpy.dot(tmp, input_vector.T)

    return wih, who

def identify(wih, who, input_vector):
    input_vector = numpy.array(input_vector, ndmin=2).T

    output_vector = numpy.dot(wih, input_vector)
    output_vector = sigmoid(output_vector)
    
    output_vector = numpy.dot(who, output_vector)
    output_vector = sigmoid(output_vector)

    return output_vector

image_size = 28 # Specified for the MNIST Dataset
no_of_different_labels = 10 # The Numbers from 0 to 9
image_pixels = image_size * image_size
data_path = "C:/Users/HP/Desktop/Athul/Academics B.Tech/Semester 6/Introduction to Data Communication/Mini Project - Digit Recognition/"

# MNIST Dataset has been pickled to make access faster!!!!
with open(data_path + "pickled_mnist.pkl", "br") as fh:
	data = pickle.load(fh)

train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

no_of_in_nodes = image_pixels
no_of_out_nodes = 10 
no_of_hidden_nodes = 100
learning_rate = 0.2

#Create Weighted Matrix
wih, who = create_weight_matrices(no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes) #Create a Matrix to store the weights between Nodes

# Train the Neural Network (Assign Values to Weighted Matrix using the whole MNIST Dataset)
print("Initiating Training...")
for i in range(int(len(train_imgs))):
	wih, who = train(wih, who, learning_rate, train_imgs[i], train_labels_one_hot[i])
print("Dataset has been Trained!!!")

with open(data_path + "neural_net.pkl", "bw") as line:
    data = (wih, 
            who)
    pickle.dump(data, line)
print("Neural Net Pickled!!!!")


#Test the Neural Network using MNIST Dataset
count = 0;
max_right = 0
max_wrong = 0
min_right = 1
min_wrong = 1
for i in range(len(test_imgs)):
	result = identify(wih, who, test_imgs[i])
	if(int(test_labels[i][0]) != int(numpy.argmax(result))):
		#if(numpy.max(result) > 0.8):
		#	print(test_labels[i][0], numpy.argmax(result), numpy.max(result))
		if(max_wrong < numpy.max(result)):
			max_wrong = numpy.max(result)
		if(min_wrong > numpy.max(result)):
			min_wrong = numpy.max(result)
	else:
		count += 1;
		#if(numpy.max(result) < 0.2):
		#	print(test_labels[i][0], numpy.argmax(result), numpy.max(result))
		if(max_right < numpy.max(result)):
			max_right = numpy.max(result)
		if(min_right > numpy.max(result)):
			min_right = numpy.max(result)

accuracy = (count / len(test_imgs) ) * 100.0
print("Accuracy from testing with " + str(len(test_imgs)) + " pictures = " + str(accuracy) + "%")
max_right *= 100
max_wrong *= 100
min_right *= 100
min_wrong *= 100
print("Highest Accuracy of Right matches = " + str(max_right) + "%")
print("Highest Accuracy of Wrong matches = " + str(max_wrong) + "%")
print("Lowest Accuracy of Right matches = " + str(min_right) + "%")
print("Lowest Accuracy of Wrong matches = " + str(min_wrong) + "%")
