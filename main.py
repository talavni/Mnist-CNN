import numpy as np

from keras.datasets.mnist import load_data
from keras.models import Model
from keras.layers import Conv2D, Input, Activation
from keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = load_data()

train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

x_train /= 255
x_test /= 255

a = Input(shape=(x_train.shape[0], 28, 28, 1))
b = Conv2D(10, (3,3), padding='same', activation='relu')(a)
c = Conv2D(10, (3,3), padding='same', activation='relu')(b)
d = Conv2D(10, (3,3), padding='same', activation='relu')(c)
e = Conv2D(3, (1,1), activation='relu')(d)
model = Model(Inputs=a, Outputs=e)
model.compile(optimizer=Adam(beta_2=0.9), loss='mean_squared_error')
model.fit(x=x_train, y=y_train, epochs=10)
model.evaluate(x_test, y_test)

model.save_weights("weights_to_cnn")
