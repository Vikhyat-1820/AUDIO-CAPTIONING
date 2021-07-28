
## import modules and libraries


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input,Embedding,Dense
from tensorflow.keras.layers import Flatten,Conv2D,MaxPooling2D,concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional
import sys

sys.path.append("C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/utils")
sys.path.append("C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Data_Handler")

from file_io import load_yaml_file,update_settings,save_pkl_file,load_pkl_file
from Dataset_creation import dataset_creation

print("Files Loaded")





#load all settings


#main setting
main_settings_dir="C:/Users/Vikhyat/Desktop/ML/audio captioning/AUDIO CAPTIONING/Settings/dir_and_files.yaml"
main_settings=load_yaml_file(main_settings_dir)


#dataset creation setting
dataset_settings=load_yaml_file(main_settings['data_set_creation'])


#text preprocessing setting
text_settings=load_yaml_file(main_settings['text_proc_settings'])


#audio settings
audio_settings=load_yaml_file(main_settings['audio_proc_settings'])

print("Settings Loaded")


#nframes of audio features
nframes=audio_settings['nframes']
# print(nframes)

#feature type MFCC/MEL spectogram
ftype=audio_settings['feature_type']
# print(ftype)



nfrts=None

if ftype=="mel":
	nfrts=audio_settings[ftype]['nb_mels']

if ftype=="mfcc":
	nfrts=audio_settings[ftype]['n_mfcc']


# print(nfrts)


#vocab size from text settings
vocab_size=text_settings['vocab_size']


#maxlen from text settings
maxlen=text_settings['maxlen']




print("Hyper parameters loaded")

###########################################################
#######                                             #######
#######              ENCODER FOR AUDIO              #######
#######                                             #######
###########################################################


inp1=Input(shape=(nframes,nfrts,1))
conv1=Conv2D(filters=16, kernel_size=3,activation='relu')(inp1)
mp1=MaxPooling2D(pool_size=2,strides=2)(conv1)

conv2=Conv2D(filters=16, kernel_size=3,activation='relu')(mp1)
mp2=MaxPooling2D(pool_size=2,strides=2)(conv2)

conv3=Conv2D(filters=32, kernel_size=3,activation='relu')(mp2)
mp3=MaxPooling2D(pool_size=2,strides=2)(conv3)

conv4=Conv2D(filters=32, kernel_size=3,activation='relu')(mp3)
mp4=MaxPooling2D(pool_size=2,strides=2)(conv4)

conv5=Conv2D(filters=32, kernel_size=3,activation='relu')(mp4)
mp5=MaxPooling2D(pool_size=2,strides=2)(conv5)

flt1=Flatten()(mp5)
dense1=Dense(256, activation='relu')(flt1)


print("Encoder for audio created")

###########################################################
#######                                             #######
#######              ENCODER FOR TEXT               #######
#######                                             #######
###########################################################


inp2=Input(shape=(maxlen,))
emb1=Embedding(vocab_size, 256, mask_zero=True)(inp2)
lstm1=LSTM(256)(emb1)


print("Encoder for text created")

###########################################################
#######                                             #######
#######           COMBINING BOTH FEATURES           #######
#######                                             #######
###########################################################


comb1=concatenate([dense1,lstm1])


###########################################################
#######                                             #######
#######                 FINAL MODEL                 #######
#######                                             #######
###########################################################


comb2=Dense(256,activation="relu")(comb1)

oup=Dense(vocab_size,activation="softmax")(comb2)

model=Model(inputs=[inp1,inp2],outputs=oup)





model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

print(model.summary())



print("Model created")

for i in range(1):
	model.fit_generator(dataset_creation(50),epochs=1)



print("Model trained")


model.save('MODEL.h5')


