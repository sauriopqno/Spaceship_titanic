#Spaceship_titanic
# spaceship titanic with keras and deep learning 
# Application to the kaggle competition by Imanol Elizondo Perucich.

# %%
#import libraries
import tensorflow as tf 
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing

# %%
#read the dataset with pandas
data=pd.read_csv("C:\\Users\\elizo\\Desktop\\proyectos\\spacetitan\\train.csv") #path to files
test=pd.read_csv("C:\\Users\\elizo\\Desktop\\proyectos\\spacetitan\\test.csv") 
testid=test["PassengerId"] #save this column


# %%
#eliminate the name column and fill in the missing values
def clean(data):
    data=data.drop(["Name"], axis=1)
    cols=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","Age"]
    for col in cols:
        data[col].fillna(data[col].median(),inplace=True)
    data.HomePlanet.fillna("uknown", inplace=True)
    data.CryoSleep.fillna(False, inplace=True)
    data.Destination.fillna("uknown", inplace=True)
    data.VIP.fillna(False, inplace=True)
    data.Cabin.fillna("u/0/u", inplace=True)

    return data



# %%
# The cabin column has 3 pieces of data: deck, number and side, so we created an individual column for each.
def cabin(dat):
    a=[]
    b=[]
    c=[]
    for i in (range(len(dat["Cabin"]))):
        a.append(dat["Cabin"][i].split("/")[0])
        b.append(dat["Cabin"][i].split("/")[1])
        c.append(dat["Cabin"][i].split("/")[2])
    dat["cabin0"]=a
    dat["cabin1"]=b
    dat["cabin2"]=c
    return dat
    


# %%
#The second part of the passengerid column gives us the number of people within a group so we extract the ones that are together in another column
def pasid(dat):
    x=[]
    for i in range (len(dat["PassengerId"])):
       if(i+2>len(dat["PassengerId"])):
           x.append(0)
       else:
            if dat["PassengerId"][i].split("_")[1]=="01" and dat["PassengerId"][i+1].split("_")[1]=="01" :
                x.append(0)
            else:
                x.append(1)
    
    dat["family"]=x
    return dat

# %%
#now we extract the number of people in each grup
def group(dat):
    x=[]
    for i in range (len(dat["PassengerId"])):
       x.append(int(dat["PassengerId"][i].split("_")[0]))
    
    dat["groupn"]=x
    return dat

# %%
#we run the functions for the test and data 
data=clean(data)
test=clean(test)
data=cabin(data)
test=cabin(test)
data=pasid(data)
test=pasid(test)
data=group(data)
test=group(test)
#we delete the extra columns 
data=data.drop(["PassengerId"], axis=1)
test=test.drop(["PassengerId"], axis=1)
data=data.drop(["Cabin"], axis=1)
test=test.drop(["Cabin"], axis=1)




# %%
#We transform text data into numbers
le=preprocessing.LabelEncoder()
cols=["HomePlanet","CryoSleep","Destination","VIP","cabin0","cabin2"]
for col in cols:
    data[col]=le.fit_transform(data[col])
    test[col]=le.transform(test[col])
    
data.head()
    

# %%
data["Transported"]=le.fit_transform(data["Transported"])

# %%
#we select our y and x and transform them into numpy arrays 
y =data ["Transported"]
x= data.drop("Transported",axis=1)
x = np.asarray(x).astype(np.float32)
y = np.asarray(y).astype(np.float32)
 

# %%
x.shape

# %%
#we build our model with layers
model=tf.keras.Sequential([
    keras.layers.InputLayer(input_shape=(15)),
    keras.layers.BatchNormalization(input_shape=(x.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout regularization
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),  # Dropout regularization
    
    keras.layers.Dense(1, activation='sigmoid')
    
    
    
    ])
#we compile our model 
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])


# %%
model.summary()

# %%
#we train our model with a 20% validation split and 200 epochs
model.fit(x,y,validation_split=.2, epochs=200 ,batch_size=32, verbose=1)

# %%
#transform the test data into a numpy array and predict the results 
test = np.asarray(test).astype(np.float32)
pred=model.predict(test)
binary_predictions = (pred > 0.5).astype(int).flatten()


# %%
#make a pandas dataframes with the results 
df=pd.DataFrame({"PassengerId":testid.values, "Transported" : binary_predictions})

# %%
#Our data is in numeric values ​​so we transform it into boolean values.
for i in range(len(df["Transported"])):
    if (df["Transported"][i]==0):
        df["Transported"][i]=False
    else:
        df["Transported"][i]=True

# %%
print(df)

# %%
#we make a csv file with the final results 
df.to_csv("submisions.csv",index=False)


