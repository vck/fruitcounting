import glob
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
import matplotlib.pyplot as plt

D1 = glob.glob('buah/*')
D2 = glob.glob('nonbuah/*')

data_path = D1+D2

data = []

n_row, n_col = 2, 5

for image_path in data_path:
    im = Image.open(image_path)
    data.append(im.histogram())

def get_label(path):
    return path.split('/')[-1].split('-')[0]


def print_score(x, y):
    print('{}%'.format(model.score(x_, y_)*100))


data = np.array(data)
target = np.array([get_label(path) for path in data_path])
target = target.reshape(target.shape[0], 1)

x, x_, y, y_ = train_test_split(data, target, random_state=100)


model = LogisticRegression()
model1=DecisionTreeClassifier()
model.fit(x, y)
predicted = model.predict(x_)

for i in range(x_.shape[0]):
    print("ground truth", y_[i], "predicted", predicted[i])

print_score(x_, y_)
    

def print_digits(images, y, max_n=10):
    # set up the figure size in inches
    fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    i=0
    while i < max_n and i < len(image_path):
        p = fig.add_subplot(n_row, n_col, i + 1, xticks=[], yticks=[])
        image = Image.open(images[i])
        p.imshow(image, cmap=plt.cm.bone, interpolation='nearest')
        # label the image with the target value
        p.text(0, -1, str(y[i]))
        i = i + 1


    
print_digits(data_path, predicted, max_n=10)
plt.show()