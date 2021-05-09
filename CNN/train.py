from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json




data = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = data[:,0:8]
y = data[:,8]
print(x)


model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
print('\n\nadded layers\n\n')


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x,y,epochs=10,batch_size = 15)



_, accuracy = model.evaluate(x,y)
print('\n\nAccuracy = %.2f \n\n' % (accuracy*100))


model_json = model.to_json()
with open("model.json" ,"w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("\n\nSaved model to disk\n\n")

