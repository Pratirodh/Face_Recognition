from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle

model_location = './models/model1.pkl'

def train():
    # load faces
    data = load('D:/venv/dataset.npz')
    testX_faces = data['arr_2']
    # load face embeddings
    data = load('D:/venv/faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    with open(model_location,'wb') as f:
        pickle.dump(model,f)
    
if __name__ == "__main__":
    train() 

