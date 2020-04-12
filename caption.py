from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from keras.models import load_model
import numpy as np
import pickle
model = ResNet50(weights="imagenet",input_shape=(224,224,3))
new_model = Model(model.input,model.layers[-2].output)
new_model._make_predict_function()
with open('term_dict.pickle', 'rb') as handle:
    vocabulary = pickle.load(handle)
model = load_model("model_6.h5")
model._make_predict_function()
glove_embedding = {}
with open("glove.6B.50d.txt",'r',encoding="utf8") as f:
    for i in f:
        ax = i.split()
        glove_embedding[ax[0]]=np.array(ax[1:],dtype='float')
embedding_dim = 50

import numpy as np
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
def encode_image(img_path):
    img = preprocess_img(img_path)
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector

max_len = 35
from keras.preprocessing import sequence
id_word = dict([(val,key) for (key,val) in vocabulary.items()])
def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequen = [vocabulary[w] for w in in_text.split() if w in vocabulary]
        sequen = sequence.pad_sequences([sequen],maxlen=max_len,value=0,padding = 'post')
        
        ypred = model.predict([photo,sequen])
        ypred = ypred.argmax()
        word = id_word[ypred]
        in_text += ' '+word
        if word=='endseq':
            break
            
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption

def Caption(img_path):
    return predict_caption(encode_image(img_path).reshape((1,2048)))

if __name__ == "__main__":
    print(Caption("picpk.jpg"))