import keras
from keras.models import Model
from keras.layers import Input,Dropout,Dense,AveragePooling2D,Conv2D,MaxPooling2D,Flatten


# def encoder():
model=keras.Sequential()
model.add(Input(shape=(600,128,1)))
model.add(Conv2D(filters=16, kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Conv2D(filters=16, kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Conv2D(filters=32, kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Conv2D(filters=32, kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Conv2D(filters=32, kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

print(model.summary())

	# inp=Input(shape=(600,128,))
	# conv1=Conv2D(filters=16, kernel_size=3,activation='relu')(inp)
	# mxpool1=MaxPooling2D(pool_size=2,stride=2)(conv1)

	# conv2=Conv2D(filters=16, kernel_size=3,activation='relu')(mxpool1)
	# mxpool2=MaxPooling2D(pool_size=2,stride=2)(conv2)

	# conv3=Conv2D(filters=32, kernel_size=3,activation='relu')(mxpool2)
	# mxpool3=MaxPooling2D(pool_size=2,stride=2)(conv3)

	# conv4=Conv2D(filters=32, kernel_size=3,activation='relu')(mxpool3)
	# mxpool4=MaxPooling2D(pool_size=2,stride=2)(conv4)

	# conv5=Conv2D(filters=32, kernel_size=3,activation='relu')(mxpool4)
	# mxpool5=MaxPooling2D(pool_size=2,stride=2)(conv5)

	# flatten=Flatten()(mxpool5)

	# dense1=
	# return model


