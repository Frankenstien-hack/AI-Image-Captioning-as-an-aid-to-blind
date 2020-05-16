import cv2
import pickle
import pyttsx3
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = load_model("./Model_3_Weights/model_19.h5")
model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)

def preprocess_image(img):
    img = load_img(img, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector

with open("Model_3_index_words/words_to_index.pkl", "rb") as w2i:
    words_to_index = pickle.load(w2i)
    
with open("Model_3_index_words/index_to_words.pkl", "rb") as i2w:
    index_to_words = pickle.load(i2w)
 
def predict_caption(photo):

    input_text = "<s>"
    max_length = 35

    for i in range(max_length):
        sequence = []
        for w in input_text.split():
            if w in words_to_index:
                sequence.append(words_to_index[w])
        sequence = pad_sequences([sequence], maxlen = max_length, padding = "post")

        y_pred = model.predict([photo, sequence])
        y_pred = y_pred.argmax()
        word = index_to_words[y_pred]
        input_text += ' ' + word
        
        if word == "<e>":
            break
        
    final_caption =  input_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def caption_image(image):

    encoded_img = encode_image(image)
    caption = predict_caption(encoded_img)
    return caption

# img = "static/b.jpg"
# cap = caption_image(img)
# print(cap)
# engine = pyttsx3.init()
# cam = cv2.VideoCapture(0)
# i=0

# while True:
#     ret,frame = cam.read()
#     key_pressed = cv2.waitKey(1)&0xFF
#     if ret==False:
#         print("Something went wrong")
#         continue
#     if key_pressed == ord('q'):
#         break

#     if i%100 == 0:
#         cv2.imwrite("static/"+str(i)+".jpg", frame)       
#         cap = caption_image("static/"+str(i)+".jpg")
#         print(cap) 
#         engine.say(cap)  
#         engine.runAndWait() 
#     i += 1
# cam.release()
# cv2.destroyAllWindows()