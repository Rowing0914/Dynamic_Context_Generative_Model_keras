import keras.backend as K
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, Reshape
from keras.optimizers import RMSprop
from tensorflow.python import debug as tf_debug

# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

x_train = np.random.rand(4,4)
y_train = np.random.rand(4,4,1)
x_test  = np.random.rand(4,4)
y_test  = np.random.rand(4,4,1)

main_input_1 = Input(shape=(4, ), name='main_input_1')
main_input_2 = Input(shape=(4, ), name='main_input_2')
# main_input__1 = Dense(4)(main_input_1)
# main_input__2 = Dense(4)(main_input_2)
# main_input = keras.layers.add([main_input__1, main_input__2])
main_input = keras.layers.add([main_input_1, main_input_2])

x = Embedding(output_dim=4, input_dim=4, input_length=4, name='embedding')(main_input)
Dense_out = Dense(4, name='dense')(x)
temp_output = Dense(1, activation='sigmoid', name='temp_output')(Dense_out)
second_input = Input(shape=(4,), name='temp_input')
x = keras.layers.add([Dense_out, second_input])
main_output = Dense(1, activation='softmax', name='dense2')(x)
model = Model(inputs=[main_input_1, main_input_2, second_input], outputs=[main_output, temp_output])
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', loss_weights=[1., 0.2], metrics=['accuracy'])
model.fit([x_train, x_train, x_train], [y_train, y_train], epochs=2)
score = model.evaluate([x_test, x_test, x_test], [y_test, y_test])
print(score[0], score[1])
