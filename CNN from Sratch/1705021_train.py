import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class CNN:
    def __init__(self):
        pass

    def forward_propagation(self, X):
        pass

    def backward_propagation(self, dL_dZ,learning_rate):
        pass
    

class SoftMax(CNN):
    def __init__(self):
       pass

    def __str__(self):
        return "SoftMax"

    def forward_propagation(self, Z):
        self.expZ = np.exp(Z)
        # self.y_hat = self.expZ / np.sum(self.expZ, axis=0) # axis = 0 bcz we want to sum along the column where each column is a sample
        self.y_hat = self.expZ /np.einsum('ij->j', self.expZ)
        # print(self.y_hat.shape)
        return self.y_hat # y_hat.shape = (num_class, batch_size)

    def backward_propagation(self, grad,learning_rate):
        # print(grad.shape)
        dL_dZ = np.copy(grad)

        return dL_dZ # dL_dZ.shape = (num_class, batch_size)


class Dense(CNN):
    def __init__(self, output_dimension):
        self.Z_dimension = output_dimension
        self.W = None
        self.B = None

    def __str__(self):
        return "Dense"

    def forward_propagation(self, X):
        self.X = X # X.shape = (input_dimension, batch_size)
        self.batch_size = X.shape[1]
     
        if self.W is None:
            # print("Initializing weights and biases at Dense layer")
            self.W = np.random.randn(self.Z_dimension, self.X.shape[0]) * math.sqrt(2 /X.shape[0]) # W.shape = (output_dimension, input_dimension)
            self.B = np.zeros((self.Z_dimension, 1))
           

        # self.Z = np.dot(self.W, self.X) + self.B
        self.Z = np.einsum('ij,jk->ik', self.W, self.X) + self.B
         
        return self.Z  # Z.shape = (output_dimension, batch_size)

    def backward_propagation(self, dL_dZ, learning_rate): # dL_dZ.shape = (output_dimension, batch_size)
        # dL_dW = dL_dZ * dZ_dW = (1/m) * dL_dZ * X.T  and jayP video deriv of backpropagation @ 19:17
        # we divide by m bcz we want to take the average of the gradients of all the samples thus summarization is done here

        # dL_dW = np.dot(dL_dZ, self.X.T)/ self.batch_size  # dL_dW.shape = (output_dimension, input_dimension)
        dL_dW = np.einsum('ij,kj->ik', dL_dZ, self.X) / self.batch_size

        # dL_dB = np.sum(dL_dZ, axis=1, keepdims=True) / self.batch_size # dL_dB = (output_dimension, 1) where axis = 1 bcz we want to sum along the row 
        dL_dB = np.einsum('ij->i', dL_dZ) / self.batch_size
        dL_dB = dL_dB.reshape(self.Z_dimension, 1)

        # dL_dX = np.dot(self.W.T, dL_dZ)  # dL_dX = dL_dZ * dZ_dX = dL_dZ * W.T
        dL_dX = np.einsum('ij,ik->jk', self.W, dL_dZ)
        


        self.W -= learning_rate * dL_dW
        self.B -= learning_rate * dL_dB
        return dL_dX  # dL_dX.shape = (input_dimension, batch_size)


class Flatten(CNN):
    def __init__(self):
        pass

    def __str__(self):
        return "Flatten"

    def forward_propagation(self, X): # X.shape = (batch_size, height, width, num_channels)
        self.X_shape = X.shape
        copy = np.copy(X)
        Z = np.reshape(copy, (X.shape[0],np.prod(X.shape[1:], dtype=int)))

        # Z = Z.T
        Z = np.einsum("ij -> ji", Z)
        # Z.shape = (num_features, batch_size) where num_features is 1st axis bcz we need individual column per sample image
        return Z

    def backward_propagation(self, dL_dZ, learning_rate): # dL_dZ.shape = (num_features, batch_size)
        dL_dX = np.einsum("ij -> ji", dL_dZ)
        dL_dX =  dL_dX.reshape(self.X_shape) # dL_dX.shape = (batch_size, height, width, num_channels)
        return dL_dX


class ReLuActivation(CNN):
    def __init__(self):
        pass

    def __str__(self):
        return "ReLU"

    def forward_propagation(self, X):
        self.X = X
        Z = np.maximum(X, 0)
        return Z
        
    def backward_propagation(self, dL_dZ, learning_rate):
        dL_dX = dL_dZ * (self.X > 0)  # if Z = Relu(X) then dL_dX = dL_dZ * dZ_dX where   dZ/dX = 1 if X_mn > 0 else 0
        return dL_dX


class MaxPooling(CNN): 
     
    def __init__(self, filter_dimension,stride):
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.masks = {}
    
    def __str__(self):
            return "MaxPooling"


    def forward_propagation(self, X):

        self.X = X
        self.batch_size, height, width, num_channels = X.shape

        self.Z_height = (height - self.filter_dimension) // self.stride + 1
        self.Z_width = (width - self.filter_dimension) // self.stride + 1

        self.Z = np.zeros((self.batch_size,  self.Z_height,  self.Z_width, num_channels))

        for i in range(self.Z_height):
            ii = i * self.stride
            for j in range(self.Z_width):
                jj = j * self.stride
                X_slice = X[:, ii : ii+self.filter_dimension, jj : jj+self.filter_dimension, :]

                self.save_mask(X_slice,(i, j))
                self.Z[:, i, j, :] = np.amax(X_slice, axis=(1, 2)) 
                # as each image  needs to be max pooled along same area of the image for each filter

        return self.Z

    def save_mask(self, x, cords):
        """"
         The values of n_idx and c_idx determine the sample and channel indices respectively, 
         while the values of the flattened indices idx determine the row and column indices. 
         The result is a 3D array where each element is 1 if the corresponding element in x is the maximum value in its row, 
         and 0 otherwise.
         """	
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self.masks[cords] = mask


    def backward_propagation(self, dL_dZ, learning_rate):
        dL_dX = np.zeros_like(self.X)
        height,width = dL_dZ.shape[1:3]

        for i in range(height):
            ii = i * self.stride
            for j in range(width):
                jj = j * self.stride
                dL_dX[:, ii:ii+self.filter_dimension, jj:jj+self.filter_dimension, :] += dL_dZ[:, i:i+1, j:j+1, :] * self.masks[(i, j)]
        return dL_dX    



class Convolution(CNN):
    def __init__(self, output_channels, filter_dimension, stride,padding):
        self.num_filters = output_channels
        self.filter_dimension = filter_dimension
        self.padding = padding
        self.stride = stride
        self.flag = 0



    def __str__(self):
        return "Convolution"

    def forward_propagation(self, X):
        self.X = X
        batch_size, height, width, num_channels = X.shape
        self.Z_height = int((height - self.filter_dimension+ 2 * self.padding)/self.stride) + 1
        self.Z_width = int((width - self.filter_dimension+ 2 * self.padding)/self.stride) + 1
        self.Z = np.zeros((batch_size, self.Z_height, self.Z_width, self.num_filters))
        if self.flag == 0:
            # print("Initializing weights and biases from conv")
            self.W = np.random.randn(self.num_filters, self.filter_dimension, self.filter_dimension, num_channels)*  math.sqrt(2 / (self.filter_dimension * self.filter_dimension * num_channels))
            self.B = np.zeros(self.num_filters)
            self.flag = 1

        self.X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, num_channels))
        self.X_padded[:, self.padding: self.padding + height, self.padding: self.padding + width, :] = self.X


        
        for k in range(batch_size):
            for l in range(self.num_filters): # value from different filters convolved to the same channel
                for i in range(self.Z_height):
                    for j in range(self.Z_width):
                        self.Z[k, i, j, l] = np.sum(self.X_padded[k, i * self.stride: i * self.stride + self.filter_dimension, j * self.stride: j * self.stride + self.filter_dimension, :] * self.W[l]) + self.B[l]

        return self.Z

    def backward_propagation(self, dL_dZ, learning_rate):

        num_samples = dL_dZ.shape[0]
        input_dim = dL_dZ.shape[1]
        input_dim_pad = (input_dim - 1) * self.stride + 1         ## 
        output_dim = self.X_padded.shape[1] - 2 * self.padding
        num_channels = self.X_padded.shape[3]
        
        del_b = np.sum(dL_dZ, axis=(0, 1, 2)) / num_samples
        
        del_v_sparse = np.zeros((num_samples, input_dim_pad, input_dim_pad, self.num_filters))
        del_v_sparse[:, :: self.stride, :: self.stride, :] = dL_dZ
       
        del_w = np.zeros((self.num_filters, self.filter_dimension, self.filter_dimension, num_channels))
        for l in range(self.num_filters):
            for i in range(self.filter_dimension):
                for j in range(self.filter_dimension):
                    del_w[l, i, j, :] = np.mean(np.sum(self.X_padded[:, i: i + input_dim_pad, j: j + input_dim_pad, :] * np.reshape(del_v_sparse[:, :, :, l], del_v_sparse.shape[: 3] + (1,)), axis=(1, 2)), axis=0)
        
       

        del_v_sparse_pad = np.pad(del_v_sparse, ((0,), (self.filter_dimension - 1 - self.padding,), (self.filter_dimension - 1 - self.padding,), (0,)), mode='constant')
        weights_prime = np.rot90(np.transpose(self.W, (3, 1, 2, 0)), 2, axes=(1, 2))

        dL_dX = np.zeros((num_samples, output_dim, output_dim, num_channels))
        for k in range(num_samples):
            for l in range(num_channels):
                for i in range(output_dim):
                    for j in range(output_dim):
                        dL_dX[k, i, j, l] = np.sum(del_v_sparse_pad[k, i: i + self.filter_dimension, j: j + self.filter_dimension, :] * weights_prime[l])
        


        self.W -= learning_rate * del_w
        self.B -= learning_rate * del_b

        return dL_dX

class Toy:
    def __init__(self):
        pass

    def get_toy(self,one_hot=False):
        file_dir = 'Toy Dataset'
        
        train_ds = pd.read_csv(f'{file_dir}/trainNN.txt', sep=r'\s+', header=None)
        train_data = np.vstack((train_ds[0], train_ds[1], train_ds[2], train_ds[3])).T
        train_labels = train_ds[4].to_numpy()

        test_ds = pd.read_csv(f'{file_dir}/testNN.txt', sep=r'\s+', header=None)
        test_data = np.vstack((test_ds[0], test_ds[1], test_ds[2], test_ds[3])).T
        test_labels = test_ds[4].to_numpy()

        toy = {}
        toy['train_data'] = train_data
        toy['train_labels'] = train_labels
        toy['test_data'] = test_data
        toy['test_labels'] = test_labels

        if one_hot:
            from sklearn.preprocessing import OneHotEncoder
        
            enc = OneHotEncoder()
            one_hot_encoded = enc.fit_transform(toy['train_labels'].reshape(-1, 1)).toarray()
            toy["train_one_hot"] = one_hot_encoded

            one_hot_encoded = enc.fit_transform(toy['test_labels'].reshape(-1, 1)).toarray()
            toy["test_one_hot"] = one_hot_encoded
    

        return toy


    def train_toy(self):
        toy = self.get_toy(one_hot=True)
        X = toy["train_data"]
        Y = toy["train_one_hot"]
        test_X = toy["test_data"]
        test_Y = toy["test_one_hot"]

        cnn_model = Model('./model1.txt')
        cnn_model.train_toy(X, Y, test_X, test_Y, batch_size=30, n_epochs=25, learning_rate=1e-4)


class Model:
    def __init__(self, file_name= None):
        
        self.model_layers = []
        self.statistics = {'train_loss':[],
                           'train_acc':[],
                           'train_f1':[],
                           'val_loss':[],
                           'val_acc':[],
                           'val_f1':[]}


        if file_name is not None:
            model_file = open("./"+file_name, 'r')
            Lines = model_file.readlines()
            
            for line in Lines:
                command = line.split()
                if command[0] == 'Conv':
                    self.model_layers.append(Convolution(output_channels=int(command[1]), filter_dimension=int(command[2]), stride=int(command[3]), padding=int(command[4])))
                elif command[0] == 'ReLU':
                    self.model_layers.append(ReLuActivation())
                elif command[0] == 'MaxPool':
                    self.model_layers.append(MaxPooling(filter_dimension=int(command[1]), stride=int(command[2])))
                elif command[0] == 'Flatten':
                    self.model_layers.append(Flatten())
                elif command[0] == 'Dense':
                    self.model_layers.append(Dense(output_dimension=int(command[1])))
                elif command[0] == 'SoftMax':
                    self.model_layers.append(SoftMax())


    def generate_statistics(self, outfile=None):

        epochs = len(self.statistics['train_loss'])
        
        if outfile is not None:
            with open(outfile, 'w') as f:
                f.write('Epoch   Train Loss   Train Accuracy   Train F1   Valid Loss   Valid Accuracy   Valid F1 \n')
                for i in range(epochs):
                    f.write(f'{i+1}   {self.statistics["train_loss"][i]}   {self.statistics["train_acc"][i]}   {self.statistics["train_f1"][i]}   {self.statistics["val_loss"][i]}   {self.statistics["val_acc"][i]}   {self.statistics["val_f1"][i]} \n')
                f.close()    
        else:
            print('Epoch   Train Loss   Train Accuracy   Train F1   Valid Loss   Valid Accuracy   Valid F1')
            for i in range(epochs):
                print(f'{i+1}   {self.statistics["train_loss"][i]}   {self.statistics["train_acc"][i]}   {self.statistics["train_f1"][i]}   {self.statistics["val_loss"][i]}   {self.statistics["val_acc"][i]}   {self.statistics["val_f1"][i]}')
        
        self.plot_statistics()
        
    def plot_statistics(self):
        epochs = len(self.statistics['train_loss'])
        x = np.arange(1, epochs+1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, self.statistics['train_loss'], label='Training Loss')
        plt.plot(x, self.statistics['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Loss.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(x, self.statistics['train_acc'], label='Training Accuracy')
        plt.plot(x, self.statistics['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('Accuracy.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(x, self.statistics['train_f1'], label='Training F1')
        plt.plot(x, self.statistics['val_f1'], label='Validation F1')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.legend()
        plt.savefig('F1.png')
        plt.show()




    def get_weights(self):
        weights = []
        for layer in self.model_layers:
            if isinstance(layer, Convolution):
                weights.append(layer.W)
                weights.append(layer.B)
            elif isinstance(layer, Dense):
                weights.append(layer.W)
                weights.append(layer.B)
        return weights   


    def set_weights(self, weights):
        count = 0
        for layer in self.model_layers:
            if isinstance(layer, Convolution):
                layer.W = weights[count]
                layer.B = weights[count+1]
                count += 2
            elif isinstance(layer, Dense):
                layer.W = weights[count]
                layer.B = weights[count+1]
                count += 2          
   
    def add_layer(self, layer):
        self.model_layers.append(layer)

    

    def calculate_loss(self, y, y_hat):
     
        # print(np.sum(y_hat, axis=1))
        # print(np.argmax(y_hat, axis=1))
        # print(np.argmax(y, axis=1))
        # count = 0
        # for i in range(y_hat.shape[0]):
        #     if np.argmax(y_hat[i]) == np.argmax(y[i]):
        #         count += 1
        # print("Accuracy: ", count)
        ln_y_hat = np.log(y_hat)
        out = - np.sum( y * ln_y_hat,axis=1)
        return out
 
    def forward(self, x):
        for i in range(len(self.model_layers)):
            # print(x.shape)
            # print(self.model_layers[i])
            x = self.model_layers[i].forward_propagation(x)
        
        return x
    def backward(self, x, learning_rate):    
      
        for i in reversed(range(len(self.model_layers))):
            x = self.model_layers[i].backward_propagation(x, learning_rate)
        

    def _shuffle(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        p = np.random.permutation(X.shape[0])
        return X[p], Y[p]

    def reverse_one_hot_encoding(self,y):
        yy = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            yy[i] = np.argmax(y[i])
 
        return yy.astype(int).reshape(y.shape[0], 1)
    
    def choose_random_samples(self, X, Y, n):
        assert X.shape[0] == Y.shape[0]
        p = np.random.permutation(X.shape[0])
        return X[p][:n], Y[p][:n]
   
    def evaluate(self, X, Y, batch_size):
        
        print("Validation...")
        n_batches = (Y.shape[0] + batch_size - 1) // batch_size
        # print(" batches: ", n_batches)
        
        total_acc = 0
        total_loss = 0
        total_inputs = 0

        with tqdm(total=n_batches,disable=False) as t:
                
            for batch_num in range(n_batches):
                    x = X[batch_num*batch_size:(batch_num+1)*batch_size]
                    y = Y[batch_num*batch_size:(batch_num+1)*batch_size]
                    # print(f'(Training) Epoch: {epoch + 1} -> {batch_num + 1}/{n_batches} Batches Trained.', end='\r')
                    
                    yy_hat = self.forward(x)
                    y_hat  = np.einsum('ij->ji', yy_hat) 
    
                    losses = self.calculate_loss(y, y_hat)

                    # metrics
                    total_acc += np.sum(np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1)).astype(int))
                    total_loss += np.sum(losses)
                    valid_f1 =  f1_score(np.argmax(y, axis=1), np.argmax(y_hat, axis=1), average='macro')
                    total_inputs += losses.shape[0]
                
                    t.set_postfix({
                        'loss': total_loss / total_inputs,
                        'acc': total_acc * 100/ total_inputs,
                        'f1': valid_f1 
                    })
                    t.update(1)
            
            self.statistics['val_loss'].append(total_loss / total_inputs)
            self.statistics['val_acc'].append(total_acc * 100/ total_inputs)
            self.statistics['val_f1'].append(valid_f1)  
           



    def predict(self, X, batch_size=128):
        
        print("Prediction...")
        n_batches = (X.shape[0] + batch_size - 1) // batch_size
        # print(" batches: ", n_batches)
        predicted_labels = np.zeros(X.shape[0])
        with tqdm(total=n_batches) as t:
                
            for batch_num in range(n_batches):
                x = X[batch_num*batch_size:(batch_num+1)*batch_size]
                
                yy_hat = self.forward(x)
                y_hat = yy_hat.T

                for i in range(y_hat.shape[0]):
                    predicted_labels [batch_num*batch_size + i] = np.argmax(y_hat[i])
                    
                t.update(1)

        return predicted_labels

    def train(self, train_X, train_Y, val_X, val_Y, batch_size, n_epochs, learning_rate):
    
        n_batches = (train_Y.shape[0] + batch_size - 1) // batch_size
       
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}")
            with tqdm(total=n_batches,disable=False) as t:
                
                total_acc = 0
                total_loss = 0
                total_inputs = 0
                
                shuffled_X, shuffled_Y = self._shuffle(train_X, train_Y)

                for batch_num in range(n_batches):
                    x = shuffled_X[batch_num*batch_size:(batch_num+1)*batch_size]
                    y = shuffled_Y[batch_num*batch_size:(batch_num+1)*batch_size]
                    # print(f'(Training) Epoch: {epoch + 1} -> {batch_num + 1}/{n_batches} Batches Trained.', end='\r')
                    

                    yy_hat = self.forward(x)
                    y_hat  = yy_hat.T
                    dL_dy_hat = y_hat - y  # Saifur sir's CNN BACK PROPAGATION pdf page 5
                    # print("dl_dy_hat", dL_dy_hat[0])
                    
                    losses = self.calculate_loss(y, y_hat)
                    self.backward(dL_dy_hat.T, learning_rate)

                    # metrics
                    total_acc += np.sum(np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1)).astype(int))
                    total_loss += np.sum(losses)
                    train_f1 =  f1_score(np.argmax(y, axis=1), np.argmax(y_hat, axis=1), average='macro')
                    total_inputs += losses.shape[0]
                
                    t.set_postfix({
                        'loss': total_loss / total_inputs,
                        'acc': total_acc * 100/ total_inputs,
                        'f1': train_f1
                    })
                    t.update(1)
            
            self.statistics['train_loss'].append(total_loss / total_inputs)	
            self.statistics['train_acc'].append(total_acc * 100/ total_inputs)
            self.statistics['train_f1'].append(train_f1)


            # validation
            self.evaluate(val_X, val_Y,batch_size)
             
  
    def train_toy(self, train_X, train_Y, val_X, val_Y, batch_size, n_epochs, learning_rate):
      
        n_batches = (train_Y.shape[0] + batch_size - 1) // batch_size

        for epoch in range(n_epochs):
            print(f"epoch {epoch+1}")

            with tqdm(total=n_batches) as t:
                
                total_corr = 0
                total_inputs = 0
                total_loss = 0
                
                shuffled_X, shuffled_Y = self._shuffle(train_X, train_Y)
                
                for batch_num in range(n_batches):
                    x = shuffled_X[batch_num*batch_size:(batch_num+1)*batch_size]
                    y = shuffled_Y[batch_num*batch_size:(batch_num+1)*batch_size]
                    # print(f'(Training) Epoch: {epoch + 1} -> {batch_num + 1}/{n_batches} Batches Trained.', end='\r')
                    

                    yy_hat = self.forward(x.T)
                    y_hat = yy_hat.T
                    dL_dy_hat = y_hat - y  # Saifur sir's CNN BACK PROPAGATION pdf page 5

                    # print("dl_dy_hat", dL_dy_hat[0])
                    
                    losses = self.calculate_loss(y, y_hat)
                    self.backward(dL_dy_hat.T, learning_rate)

        
                    # metrics
                    batch_corr = np.sum(np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1)).astype(int))
                    total_corr += batch_corr
                    total_loss += np.sum(losses)
                    total_inputs += losses.shape[0]
                    f1_train = f1_score(np.argmax(y, axis=1), np.argmax(y_hat, axis=1), average='macro')
                    
                    t.set_postfix({
                        'loss': total_loss / total_inputs,
                        'acc': total_corr * 100 / total_inputs,
                        'f1': f1_train
                    })
                    t.update(1)

            # validation
            self.evaluate_toy(val_X, val_Y,batch_size=batch_size)
             
    def evaluate_toy(self, X, Y, batch_size=64):

        n_batches = (Y.shape[0] + batch_size - 1) // batch_size
        # print(" batches: ", n_batches)
        total_corr = 0
        total_inputs = 0
        total_loss = 0

        with tqdm(total=n_batches) as t:
                
                for batch_num in range(n_batches):
                    x = X[batch_num*batch_size:(batch_num+1)*batch_size]
                    y = Y[batch_num*batch_size:(batch_num+1)*batch_size]

                    yy_hat = self.forward(x.T)
                    y_hat = yy_hat.T
                
                    losses = self.calculate_loss(y, y_hat)
                    # metrics
                    batch_corr = np.sum(np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1)).astype(int))
                    total_corr += batch_corr
                    total_loss += np.sum(losses)
                    total_inputs += losses.shape[0]
                    f1_train = f1_score(np.argmax(y, axis=1), np.argmax(y_hat, axis=1), average='macro')
                    
                    t.set_postfix({
                        'loss': total_loss / total_inputs,
                        'acc': total_corr * 100 / total_inputs,
                        'f1': f1_train
                    })
                    t.update(1)
       


def main():
    
    # np.random.seed(0)
    file_name = 'bangla_digits_raw.npz'
    x_train, y_train, x_test, y_test = np.load(file_name)['x_train'], np.load(file_name)['y_train'], np.load(file_name)['x_test'], np.load(file_name)['y_test']
  
    # print('x_train.shape: ',x_train.shape)
    # print('y_train.shape: ',y_train.shape)

    # print('x_test.shape: ',x_test.shape)
    # print('y_test.shape: ',y_test.shape)

    # y_train = y_train.astype(int)
    # y_test = y_test.astype(int)
    # y_valid = y_valid.astype(int)

    
    batch_size = 256
    num_epochs = 10
    learning_rate = 0.001
    
    # create model
    cnn_model = Model()
    cnn_model.add_layer(Convolution(output_channels=6, filter_dimension=5, stride=1, padding=0))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(MaxPooling(filter_dimension=2, stride=2))
    cnn_model.add_layer(Convolution(output_channels=16, filter_dimension=5, stride=1, padding=0))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(MaxPooling(filter_dimension=2, stride=2))
    cnn_model.add_layer(Flatten())
    cnn_model.add_layer(Dense(128))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(Dense(64))
    cnn_model.add_layer(ReLuActivation())
    cnn_model.add_layer(Dense(10))
    cnn_model.add_layer(SoftMax())

    # train_size = 24000
    valid_size = 8000
    # x_train, y_train = cnn_model.choose_random_samples(x_train, y_train, train_size)
    x_valid, y_valid = cnn_model.choose_random_samples(x_train, y_train, valid_size)

    print(x_train.shape)
    # print(x_test.shape)
    print(x_valid.shape)


    cnn_model.train(x_train, y_train, x_valid, y_valid,batch_size,num_epochs, learning_rate)

    cnn_model.generate_statistics()

    # Save model to pickle file
    model_weights = cnn_model.get_weights()
    with open('1705021_model.pkl', 'wb') as f:
        pickle.dump(model_weights, f)



    


if __name__ == "__main__":

    main()
    # Toy().train_toy()



