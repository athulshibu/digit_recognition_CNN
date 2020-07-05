import pickle
import numpy
from PIL import Image, ImageFilter

def sigmoid(x):
    return 1 / (1 + numpy.e ** -x)

def image_to_mnist_format(image_address):
    image = Image.open(image_address)
    image = image.convert('L')
    width = float(image.size[0])
    height = float(image.size[1])
    formatted_image = Image.new('L', (28, 28), (255))
    new_image = image.resize((20,20))
    formatted_image.paste(new_image, (4, 4))
    pic_array = numpy.array(formatted_image)
    pic_array = pic_array.flatten()
    
    factor = 0.99 / 255
    pic_array = 255 - pic_array
    pic_array = pic_array * factor + 0.01
    
    for i in range(len(pic_array)):
        if(pic_array[i] < 0.4):
            pic_array[i] = 0.01
    #print(pic_array)
    return(pic_array)

def identify(wih, who, input_vector):
    input_vector = numpy.array(input_vector, ndmin=2).T
    output_vector = sigmoid(numpy.dot(wih, input_vector))
    output_vector = sigmoid(numpy.dot(who, output_vector))
    return output_vector


data_path = "C:/Users/HP/Desktop/Athul/Academics B.Tech/Semester 6/Introduction to Data Communication/Mini Project - Digit Recognition/"
# Neural Net (Weighted Matrices of the Trained Neural Net) has been pickled to make access faster!!!!
with open(data_path + "neural_net.pkl", "br") as fh:
    data = pickle.load(fh)
wih = data[0]
who = data[1]

for i in range(1,10):
    number = i
    image = "C:/Users/HP/Desktop/Athul/Academics B.Tech/Semester 6/Introduction to Data Communication/Mini Project - Digit Recognition/Digits/"
    image = image + str(number) + ".jpg"
    result = identify(wih, who, image_to_mnist_format(image))
    accuracy = numpy.max(result) * 100
    print(str(number) + " predicted to be " + str(numpy.argmax(result)) + " with accuracy of " + str(accuracy) + "%")