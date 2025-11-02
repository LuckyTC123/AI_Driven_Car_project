#importing required
from utilis import *
from sklearn.model_selection import train_test_split
path = 'mydata' #i did generatet it using udacity simulator
data = importDataInfo(path)
#step 2 data visualization
data = balaceData(data,display=False)

imagespath,steering = loaddata(path,data)
print(imagespath[0],steering[0])

xtrain,xvalidation ,ytrain,yval =train_test_split(imagespath,steering,test_size=0.2,random_state=5)
print(len(xtrain))
print(len(xvalidation))

#screating the  model and printing summary
model =creatmodel()
model.summary()

history =model.fit(batchgen(xtrain,ytrain,100,1),steps_per_epoch=200,epochs=6,
                   validation_data=batchgen(xvalidation,yval,100,0),validation_steps=100)
model.save('model.h5')
print('model saved')

