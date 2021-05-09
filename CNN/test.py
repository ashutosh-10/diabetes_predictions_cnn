from numpy import loadtxt
from keras.models import model_from_json


data = loadtxt('pima-indians-diabetes.csv', delimiter = ',')
x = data[:,0:8]
y = data[:,8]


json_file  = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

print("\n\n Loaded model from disk \n\n")
predictions = model.predict_classes(x)
for i in range(5,10):
    print(' %s => %d (Original Class: %d)' % (x[i].tolist(), predictions[i], y[i] ) )
#print(f" original value:- {y[5]} \npredicted:- {predictions[5]}")