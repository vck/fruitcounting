from PIL import Image
import numpy as np 
import glob
from sklearn.model_selection import train_test_split

data1=glob.glob('buah/*')
data2=glob.glob('nonbuah/*')
data_label=data1+data2
#getting RGB
data_1=[]
for image in data1:
	image=Image.open(image)
	rgb_image=image.convert('RGB')
	r,g,b=rgb_image.getpixel((1,1))
	data_1.append((r,g,b))

data_2=[]
for image in data2:
	image=Image.open(image)
	rgb_image=image.convert('RGB')
	r,g,b=rgb_image.getpixel((1,1))
	data_2.append((r,g,b))

data__1=np.array(data_1)
data__2=np.array(data_2)

data_x=np.concatenate((data__1,data__2))

#get_label
def get_label(path):
    return path.split('/')[-1].split('-')[0]

def print_score(x, y):
    print('{}%'.format(clf.score(x_tes, y_tes)*100))

label=np.array([get_label(path) for path in data_label])

from sklearn.tree import DecisionTreeClassifier
x_train=data_x
y_train=label
x_tes=data_x[99:200]
y_tes=label[99:200]


clf=DecisionTreeClassifier()
clf.fit(x_train, y_train)
predict=clf.predict(x_tes)

print (print_score(x_tes, y_tes))

import matplotlib.pyplot as plt

image1=[]
for image in data_label:
	image=Image.open(image)
	image1.append(image)

image_prediction=list(zip(image1, predict))
for index, (image,prediction) in enumerate (image_prediction [:4]):
	plt.subplot(2,4, index + 5)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Prediction: {}'.format( prediction))



plt.show()

#print (data__1)
#print (len(data__1))
#print (data_x.shape)