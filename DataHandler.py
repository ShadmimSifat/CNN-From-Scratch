import os
import numpy as np
import glob
import pandas as pd
import cv2
import sys


#Declaring constants
FIG_WIDTH=20 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows
RESIZE_DIM=28 # The images will be resized to 28x28 pixels


def get_key(path):
        # seperates the key of an image from the filepath
        key=path.split(sep=os.sep)[-1]
        return key

def get_data(paths_img,path_label=None,resize_dim=None):
        # print(resize_dim)
        '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
        Args:
            paths_img: image filepaths
            path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
            resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
        Returns:
            X: group of images
            y: categorical true labels
        '''
        X=[] # initialize empty list for resized images
        for i,path in enumerate(paths_img):
            img=cv2.imread(path,cv2.IMREAD_COLOR) # images loaded in color (BGR)
            #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # cnahging colorspace to GRAY
            # img= cv2.bitwise_not(img) # inverting the image
            if resize_dim is not None:
                img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28
            #X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
            gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0) #unblur
            img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
            img = cv2.filter2D(img, -1, kernel)
            thresh = 200
            maxValue = 255
            #th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
            ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            X.append(img) # expand image to 28x28x1 and append to the list
            # display progress
            if i==len(paths_img)-1:
                end='\n'
            else: end='\r'
            print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
            
        X=np.array(X) # tranform list to numpy array
        if  path_label is None:
            return X
        else:
            df = pd.read_csv(path_label) # read labels
            df=df.set_index('filename') 
            y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
            y = np.eye(10)[y_label]  # transfrom integer value to categorical variable
            return X, y

def load_dataset(file_path,resize_dim):
        data_dir=os.path.join('.',file_path)
        # TRAINING DATA
        paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
        paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
        paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
        paths_train_all=paths_train_a+paths_train_b+paths_train_c

        # TESTING DATA
        paths_test_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))
        paths_test_all=paths_test_d

        # TRAINING LABELS
        path_label_train_a=os.path.join(data_dir,'training-a.csv')
        path_label_train_b=os.path.join(data_dir,'training-b.csv')
        path_label_train_c=os.path.join(data_dir,'training-c.csv')

        # TESTING LABELS
        path_label_test_d=os.path.join(data_dir,'training-d.csv')
  
        # GETTING TRAINING DATA  
        X_train_a,y_train_a=get_data(paths_train_a,path_label_train_a,resize_dim)
        X_train_b,y_train_b=get_data(paths_train_b,path_label_train_b,resize_dim)
        X_train_c,y_train_c=get_data(paths_train_c,path_label_train_c,resize_dim)

 
        X_train_all=np.concatenate((X_train_a,X_train_b,X_train_c),axis=0)
        y_train_all=np.concatenate((y_train_a,y_train_b,y_train_c),axis=0)

        # print('X_train_all.shape: ',X_train_all.shape)
        # print('y_train_all.shape: ',y_train_all.shape)


        X_test_d,y_test_d=get_data(paths_test_d,path_label_test_d,resize_dim)
        X_test_all=X_test_d
        y_test_all=y_test_d

        # print('X_test_all.shape: ',X_test_all.shape)
        # print('y_test_all.shape: ',y_test_all.shape)
     
        X_train_all = X_train_all.reshape(X_train_all.shape[0],resize_dim, resize_dim,1).astype('float32')
        X_test_all = X_test_all.reshape(X_test_all.shape[0],resize_dim, resize_dim,1).astype('float32')

        # print('X_train_all.shape: ',X_train_all.shape)
        # print('X_test_all.shape: ',X_test_all.shape)

        train_mean = np.mean(X_train_all)
        train_std = np.std(X_train_all)

        X_train_all = (X_train_all - train_mean) / train_std

        X_test_all = (X_test_all - train_mean) / train_std


        # X_train_all = X_train_all / 255
        # X_test_all = X_test_all / 255

        # X_train_all = X_train_all - 0.5
        # X_test_all = X_test_all - 0.5

       
        return X_train_all, y_train_all, X_test_all, y_test_all

def train_validation_splitter(X_train_all, y_train_all, validation_split=0.3):
        
        indices=list(range(len(X_train_all)))
        np.random.seed(42)
        np.random.shuffle(indices)
        ind=int(len(indices)*(1-validation_split))

        # TRAINING DATA
        X_train=X_train_all[indices[:ind]] 
        y_train=y_train_all[indices[:ind]]

        # VALIDATION DATA
        X_val=X_train_all[indices[-(len(indices)-ind):]] 
        y_val=y_train_all[indices[-(len(indices)-ind):]]

        return X_train, y_train, X_val, y_val

if __name__ == '__main__':

        if len(sys.argv) < 2:
            print("Please provide a file path as an argument")
            sys.exit()

        file_path = sys.argv[1] 

        file_name = 'bangla_digits_raw32.npz'
        x_train, y_train, x_test, y_test = load_dataset(file_path,resize_dim=32)
        print('x_train.shape: ',x_train.shape)
        print('y_train.shape: ',y_train.shape)
        np.savez_compressed(file_name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        print('Data saved to ', file_name)

        # x_train, y_train, x_test, y_test = np.load(file_name)['x_train'], np.load(file_name)['y_train'], np.load(file_name)['x_test'], np.load(file_name)['y_test']
        # x_train, y_train, x_val, y_val = train_validation_splitter(x_train, y_train, validation_split=0.3)

        # print('x_train.shape: ',x_train.shape)
        # print('y_train.shape: ',y_train.shape)
        # print('x_val.shape: ',x_val.shape)
        # print('y_val.shape: ',y_val.shape)
        # print('x_test.shape: ',x_test.shape)
        # print('y_test.shape: ',y_test.shape)



