import os
import numpy as np
import pandas as pd
import glob
import cv2
import sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score

puzzle = __import__('1705021_train')

Model = puzzle.Model
Convolution = puzzle.Convolution
ReLuActivation = puzzle.ReLuActivation
MaxPooling = puzzle.MaxPooling
Flatten = puzzle.Flatten
Dense = puzzle.Dense
SoftMax = puzzle.SoftMax



# Declaring constants
FIG_WIDTH = 20  # Width of figure
HEIGHT_PER_ROW = 3  # Height of each row when showing a figure which consists of multiple rows
RESIZE_DIM = 28  # The images will be resized to 28x28 pixels


def get_key(path):
    # seperates the key of an image from the filepath
    key = path.split(sep=os.sep)[-1]
    return key


def get_data(paths_img, path_label=None, resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    image_data = []
    X = []  # initialize empty list for resized images
    for i, path in enumerate(paths_img):
        # print("i : ",i," path : ",path)
        image_title = get_key(path)
        image_data.append(image_title)
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # images loaded in color (BGR)
        # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cnahging colorspace to GRAY
        if resize_dim is not None:
            img = cv2.resize(img, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)  # resize image to 28x28
        # X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)  # unblur
        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # filter
        img = cv2.filter2D(img, -1, kernel)
        thresh = 200
        maxValue = 255
        # th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        X.append(img)  # expand image to 28x28x1 and append to the list
        # display progress
        if i == len(paths_img) - 1:
            end = '\n'
        else:
            end = '\r'
        print('processed {}/{}'.format(i + 1, len(paths_img)), end=end)

    X = np.array(X)  # tranform list to numpy array
    if path_label is None:
        return X, image_data, None

    else:
        df = pd.read_csv(path_label)  # read labels
        df = df.set_index('filename')
        y_label = [df.loc[get_key(path)]['digit'] for path in paths_img]  # get the labels corresponding to the images
        y = np.eye(10)[y_label]  # transfrom integer value to categorical variable
        return X, image_data, y


def load_dataset(file_path, resize_dim=RESIZE_DIM):
    data_dir = os.path.join('.', file_path)

    # TESTING DATA
    path_test_x = glob.glob(os.path.join(data_dir, '*.png'))
    # cheak dir has a csv irrespective of the name
    path_label_y = None
    if len(glob.glob(os.path.join(data_dir, '*.csv'))) > 0:
        path_label_y = glob.glob(os.path.join(data_dir, '*.csv'))[0]

    # path_label_y = None	

    X_test, image_data, Y_test = get_data(path_test_x, path_label_y, resize_dim=resize_dim)
    # print(X_test.shape)
    X_test = X_test.reshape(X_test.shape[0], resize_dim, resize_dim, 1).astype('float32')

    # X_test = X_test / 255.0
    # X_test = X_test - 0.5

    mean = np.mean(X_test)
    std = np.std(X_test)
    X_test = (X_test - mean) / std

    return X_test, image_data, Y_test


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please provide a file path as an argument")
        sys.exit()

    file_path = sys.argv[1]

    x_test, image_data, y_test = load_dataset(file_path, resize_dim=RESIZE_DIM)
    if y_test is not None:
        np.savez_compressed('test.npz', x_test=x_test, y_test=y_test, image_data=image_data)
        x_test, y_test, image_data = np.load('test.npz')['x_test'], np.load('test.npz')['y_test'], np.load('test.npz')['image_data']
    else:
        np.savez_compressed('test.npz', x_test=x_test, image_data=image_data)
        x_test, image_data = np.load('test.npz')['x_test'], np.load('test.npz')['image_data']    

    # Load model from file
    with open('1705021_model.pkl', 'rb') as file:
        model_weights = pickle.load(file)


   # create model
    cnn_model = Model()
    cnn_model.add_layer(Convolution(output_channels=6, filter_dimension=5, stride=1, padding=0))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(MaxPooling(filter_dimension=2, stride=2))
    cnn_model.add_layer(Convolution(output_channels=16, filter_dimension=5, stride=1, padding=0))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(MaxPooling(filter_dimension=2, stride=2))
    cnn_model.add_layer(Flatten())
    cnn_model.add_layer(Dense(120))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(Dense(84))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(Dense(10))
    cnn_model.add_layer(SoftMax())

    
    cnn_model.set_weights(model_weights)


    # Test model
    predicted_labels = cnn_model.predict(x_test) # 128 is the batch size
    predicted_labels = [ int(x) for x in predicted_labels]

    if y_test is not None:
        
        y_test = np.argmax(y_test, axis=1)
        print("Actual Labels: ", y_test)
        acc_score = accuracy_score(y_test, predicted_labels)
        print("Accuracy: ", acc_score)
        macro_f1_score = f1_score(y_test, predicted_labels, average='macro') 
        print("Macro F1 Score: ", macro_f1_score)
        conf_mat = confusion_matrix(y_test, predicted_labels)
        print("Confusion Matrix: \n", conf_mat)

        # conf_file = open('1705021_confusion.txt', 'w')
        # conf_file.write(str(conf_mat))
        # conf_file.close()
    
    # print("Predicted Labels: ", predicted_labels)
    assert len(predicted_labels) == len(image_data)

    image_info = open('1705021_prediction.csv', 'w')
    image_info.write('FileName,Digit\n')
    for i in range(len(image_data)):
        image_info.write(image_data[i] + ',' + str(predicted_labels[i]) + '\n')

    image_info.close()