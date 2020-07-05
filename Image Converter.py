import numpy
from PIL import Image, ImageFilter
import pickle

data_path = "C:/Users/HP/Desktop/Athul/Academics B.Tech/Semester 6/Introduction to Data Communication/Mini Project - Digit Recognition/Digits/"
image = Image.open(data_path + '1.png')
image = image.convert('L')
width = float(image.size[0])
height = float(image.size[1])
formatted_image = Image.new('L', (28, 28), (255))

new_image = image.resize((28,28))
formatted_image.paste(new_image, (4, 4))
#formatted_image.save(data_path + 'Formatted 1.png')
pic_array = numpy.array(formatted_image)
factor = 0.99 / 255
pic_array = 255 - pic_array
pic_array = pic_array * factor + 0.01
pic_array = pic_array.flatten() 
#print(pic_array)


data_path_training = "C:/Users/HP/Desktop/Athul/Academics B.Tech/Semester 6/Introduction to Data Communication/Mini Project - Digit Recognition/"
with open(data_path_training + "pickled_mnist.pkl", "br") as fh:
	data = pickle.load(fh)
test_imgs = data[1]
min = 1.0
for i in range(len(test_imgs[0])):
	if(test_imgs[0][i] > 0.01 and test_imgs[0][i] < min):
		min = test_imgs[0][i]
print(test_imgs[0])
print(min)



