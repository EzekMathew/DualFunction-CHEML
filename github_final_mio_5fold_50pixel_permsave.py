# -*- coding: utf-8 -*-
"""Github Final MIO 5fold 50pixel permsave

"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
print(tf.__version__)
!python --version
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
!pip install rdkit-pypi -qqq
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
#from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np
import os
from google.colab import drive
drive.mount('/content/drive')
data_directory = os.path.join("/content/drive/MyDrive/")

#Load Data depending on individual directories as a pd. dataframe

data["MOL"] = data["SMILES"].apply(Chem.MolFromSmiles)

#Check the shape of dataframe
print(data.shape)

print(data.columns)

Imagesize = 50

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200


    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            vect[ idx , idy ,0] = bondorder
    return vect




mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimagezero"] = data["MOL"].apply(vectorize)

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200



    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 0] = atom.GetAtomicNum()
    return vect



mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimageone"] = data["MOL"].apply(vectorize)

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200



    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Gasteiger Charges
            charge = atom.GetProp("_GasteigerCharge")
            vect[ idx , idy, 0] = charge
    return vect



mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimagetwo"] = data["MOL"].apply(vectorize)

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200



    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Hybridization
            hyptype = atom.GetHybridization().real
            vect[ idx , idy, 0] = hyptype
    return vect



mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimagethree"] = data["MOL"].apply(vectorize)

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200



    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            vect[ idx , idy, 0] = atom.GetExplicitValence()
    return vect



mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimagefour"] = data["MOL"].apply(vectorize)

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200



    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            vect[ idx , idy, 0] = atom.GetDegree()
    return vect



mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimagefive"] = data["MOL"].apply(vectorize)

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200



    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            vect[ idx , idy, 0] = atom.GetFormalCharge()
    return vect



mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimagesix"] = data["MOL"].apply(vectorize)

##Extended Chemception
#Use these below:
def chemcepterize_mol(mol, embed=10.0, res=0.48):
    #Imagesize = 50
    #50x50

#def chemcepterize_mol(mol, embed=10.0, res=0.24):
    #Imagesize = 100
    #100x100

#def chemcepterize_mol(mol, embed=10.0, res=0.12):

    #200x200



    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            vect[ idx , idy, 0] = atom.GetIsAromatic()
    return vect



mol = data["MOL"][15]



def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimageseven"] = data["MOL"].apply(vectorize)

# Split the data into train and test

foldnumber = 1
train = data[data["fold"]!=foldnumber]

test = data[data["fold"]==foldnumber]

#Check the shape of dataframe
print(data.shape)
print(data.columns)
def format_output(data):
    #####> options include ['mGluaffinity', 'xstand', 'xnorm']. Elsewhere, this will be referenced as 'numerical'
    y1 = data.pop("norm_of_log")
    #####>
    #y1 = data.pop("Normofinvlog")
    y1 = np.array(y1)
    y2 = data.pop('ReceptorBinarized')
    y2 = np.array(y2)
    return y1, y2


train_Y = format_output(train)
test_Y = format_output(test)




def form(data):
    X0 = np.array(list(data["molimagezero"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X1 = np.array(list(data["molimageone"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X2 = np.array(list(data["molimagetwo"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X3 = np.array(list(data["molimagethree"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X4 = np.array(list(data["molimagefour"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X5 = np.array(list(data["molimagefive"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X6 = np.array(list(data["molimagesix"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X7 = np.array(list(data["molimageseven"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X0t = np.transpose(list(data["molimagezero"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X1t = np.transpose(list(data["molimageone"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X2t = np.transpose(list(data["molimagetwo"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X3t = np.transpose(list(data["molimagethree"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X4t = np.transpose(list(data["molimagefour"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X5t = np.transpose(list(data["molimagefive"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X6t = np.transpose(list(data["molimagesix"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    X7t = np.transpose(list(data["molimageseven"])).reshape(-1, Imagesize * Imagesize * 1).astype("float32")
    return X0, X1, X2, X3, X4, X5, X6, X7, X0t, X1t, X2t, X3t, X4t, X5t, X6t, X7t
train_X = form(train)
test_X = form(test)




from keras.layers import Dropout


inputshape = (Imagesize * Imagesize * 1)

input0 = Input(shape=inputshape)
#l0 = Dense(units=(Imagesize * Imagesize * 1), activation='relu')(input0)
mlp0 = Dense(units=1000, activation='relu')(input0)
mlp0 = Dropout(0.2)(mlp0)
#mlp0 = Dense(units=5, activation='relu')(mlp0)
#mlp0 = Dropout(0.3)(mlp0)

input1 = Input(shape=inputshape)
mlp1 = Dense(units=1000, activation='relu')(input1)
mlp1 = Dropout(0.2)(mlp1)
#mlp1 = Dense(units=5, activation='relu')(mlp1)
#mlp1 = Dropout(0.3)(mlp1)

input2 = Input(shape=inputshape)
mlp2 = Dense(units=1000, activation='relu')(input2)
mlp2 = Dropout(0.2)(mlp2)
#mlp2 = Dense(units=5, activation='relu')(mlp2)
#mlp2 = Dropout(0.3)(mlp2)

input3 = Input(shape=inputshape)
mlp3 = Dense(units=1000, activation='relu')(input3)
mlp3 = Dropout(0.2)(mlp3)
#mlp3 = Dense(units=5, activation='relu')(mlp3)
#mlp3 = Dropout(0.3)(mlp3)

input4 = Input(shape=inputshape)
mlp4 = Dense(units=1000, activation='relu')(input4)
mlp4 = Dropout(0.3)(mlp4)
#mlp4 = Dense(units=5, activation='relu')(mlp4)
#mlp4 = Dropout(0.3)(mlp4)

input5 = Input(shape=inputshape)
mlp5 = Dense(units=1000, activation='relu')(input5)
mlp5 = Dropout(0.3)(mlp5)
#mlp5 = Dense(units=5, activation='relu')(mlp5)
#mlp5 = Dropout(0.3)(mlp5)

input6 = Input(shape=inputshape)
mlp6 = Dense(units=1000, activation='relu')(input6)
mlp6 = Dropout(0.3)(mlp6)
#mlp6 = Dense(units=5, activation='relu')(mlp6)
#mlp6 = Dropout(0.3)(mlp6)

input7 = Input(shape=inputshape)
mlp7 = Dense(units=1000, activation='relu')(input7)
mlp7 = Dropout(0.3)(mlp7)
#mlp7 = Dense(units=5, activation='relu')(mlp7)
#mlp7 = Dropout(0.3)(mlp7)



input0t = Input(shape=inputshape)
#l0 = Dense(units=(Imagesize * Imagesize * 1), activation='relu')(input0)
mlp0t = Dense(units=1000, activation='relu')(input0t)
mlp0t = Dropout(0.2)(mlp0t)
#mlp0 = Dense(units=5, activation='relu')(mlp0)
#mlp0 = Dropout(0.3)(mlp0)

input1t = Input(shape=inputshape)
mlp1t = Dense(units=1000, activation='relu')(input1t)
mlp1t = Dropout(0.2)(mlp1t)
#mlp1 = Dense(units=5, activation='relu')(mlp1)
#mlp1 = Dropout(0.3)(mlp1)

input2t = Input(shape=inputshape)
mlp2t = Dense(units=1000, activation='relu')(input2t)
mlp2t = Dropout(0.2)(mlp2t)
#mlp2 = Dense(units=5, activation='relu')(mlp2)
#mlp2 = Dropout(0.3)(mlp2)

input3t = Input(shape=inputshape)
mlp3t = Dense(units=1000, activation='relu')(input3t)
mlp3t = Dropout(0.2)(mlp3t)
#mlp3 = Dense(units=5, activation='relu')(mlp3)
#mlp3 = Dropout(0.3)(mlp3)

input4t = Input(shape=inputshape)
mlp4t = Dense(units=1000, activation='relu')(input4t)
mlp4t = Dropout(0.3)(mlp4t)
#mlp4 = Dense(units=5, activation='relu')(mlp4)
#mlp4 = Dropout(0.3)(mlp4)

input5t = Input(shape=inputshape)
mlp5t = Dense(units=1000, activation='relu')(input5t)
mlp5t = Dropout(0.3)(mlp5t)
#mlp5 = Dense(units=5, activation='relu')(mlp5)
#mlp5 = Dropout(0.3)(mlp5)

input6t = Input(shape=inputshape)
mlp6t = Dense(units=1000, activation='relu')(input6t)
mlp6t = Dropout(0.3)(mlp6t)
#mlp6 = Dense(units=5, activation='relu')(mlp6)
#mlp6 = Dropout(0.3)(mlp6)

input7t = Input(shape=inputshape)
mlp7t = Dense(units=1000, activation='relu')(input7t)
mlp7t = Dropout(0.3)(mlp7t)
#mlp7 = Dense(units=5, activation='relu')(mlp7)
#mlp7 = Dropout(0.3)(mlp7)



# merge input models

merge = layers.concatenate([mlp0, mlp1, mlp2, mlp3, mlp4, mlp5, mlp6, mlp7, mlp0t, mlp1t, mlp2t, mlp3t, mlp4t, mlp5t, mlp6t, mlp7t])
#merge = layers.concatenate([mlp0, mlp1, mlp2, mlp3])
merge = Dense(units=2000, activation='relu')(merge)
merge = Dropout(0.1)(merge)
merge = Dense(units=500, activation='relu')(merge)

n2 = Dense(units=500, activation='relu')(merge)
n2 = Dense(units=50, activation='relu')(n2)
n2 = Dropout(0.2)(n2)
n3 = Dense(units=10, activation='relu')(n2)
y1_output = Dense(units=1, activation='sigmoid', name='numerical')(n3)


m2 = Dense(units=500, activation='relu')(merge)
m2 = Dense(units=50, activation='relu')(m2)
m2 = Dropout(0.3)(m2)
m2 = Dense(units=10, activation='relu')(m2)
y2_output = Dense(units=1, activation='sigmoid', name='ReceptorBinarized')(m2)

# Define the model with the input layer and a list of output layers
model = Model(inputs=[input0, input1, input2, input3, input4, input5, input6, input7, input0t, input1t, input2t, input3t, input4t, input5t, input6t, input7t], outputs=[y1_output, y2_output])
#model = Model(inputs=[input0, input1, input2, input3], outputs=[y1_output, y2_output])


import keras.backend as K

def customLoss(y_true,y_pred):
    return abs(((K.sqrt(y_true + 0.0001)) - (K.sqrt(y_pred + 0.0001))))/(y_true + 0.001)
    #return abs((K.sqrt(K.sqrt(y_true + 0.0001)) - K.sqrt(K.sqrt(y_pred + 0.0001))))/(K.square(y_true + 0.001))
    #return K.square(1/(y_true + 0.0001) - (1-y_pred + 0.0001))



# Specify the optimizer, and compile the model with loss functions for both outputs
#######>
optimizer = tf.keras.optimizers.Adam(lr=0.001)
#######>
model.compile(optimizer=optimizer,
              #########>

              loss={'numerical': customLoss, 'ReceptorBinarized': tf.keras.losses.BinaryCrossentropy(from_logits=False)},
              #loss={'numerical': tf.keras.losses.Huber(), 'ReceptorBinarized': tf.keras.losses.BinaryCrossentropy(from_logits=False)},

              #########>
              metrics={'numerical': tf.keras.metrics.MeanSquaredError(),
                       'ReceptorBinarized': ["accuracy"]})
tf.keras.utils.plot_model(model, "multi.png", show_shapes=True)




# Train the model
history = model.fit(train_X, train_Y,
                    epochs=100, batch_size=150, shuffle=True)
# Test the model and print loss and rmse for both outputs
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_acc = model.evaluate(test_X, test_Y)

print()
print(f'loss: {loss}')
print(f'numerical_loss: {Y1_loss}')
print(f'ReceptorBinarized_loss: {Y2_loss}')
print(f'numerical_rmse: {Y1_rmse}')
print(f'ReceptorBinarized_acc: {Y2_acc}')
def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()
# Run predict
Y_pred = model.predict(test_X)
price_pred = Y_pred[0]
ptratio_pred = Y_pred[1]

plot_diff(test_Y[0], Y_pred[0], title='numerical')
plot_diff(test_Y[1], Y_pred[1], title='ReceptorBinarized')
test_Label = test.get(["Original Label"])
test_Label = np.array(test_Label)

xxx = test_Y[0].reshape((-1,1))
yyy = Y_pred[0].reshape((-1,1))
ty = test_Y[1].reshape((-1,1))
py = Y_pred[1].reshape((-1,1))

dfx = pd.DataFrame(xxx)
dfy = pd.DataFrame(yyy)
dty = pd.DataFrame(ty)
dpy = pd.DataFrame(py)
dfz = pd.DataFrame(test_Label, columns=["Original Label"])

df = pd.concat([dfz, dfx, dfy, dty, dpy], axis=1,keys=["Original Label", "Numerical Actual", "Numerical Prediction", "Receptor Actual", "Receptor Predicted"])
print(df.shape)
df.to_excel('pixel50fold1.xlsx')
!cp pixel50fold1.xlsx "drive/My Drive/"
