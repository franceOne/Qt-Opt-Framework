import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

imgInputShape = (472,472,3)
inputsImgModel = keras.Input(shape=imgInputShape, name='img_input')

# binary open-closed gripper state = 1,  skalar height of gripper = 1
inputStateShape = (2,)

#groper pose = (translation = R³ = 3, rotation = R²= 2), open-close = onhotVector R² = 2, terminateEpisode = 1
inputActionShape= (8,)

inputStateActionShape = (10,)
inputsActionState = keras.Input(shape=inputStateActionShape, name='state_action_input')



def generateConvolutionLayer(filter, shape, input):
    x = layers.Conv2D(filter, shape, kernel_regularizer=tf.keras.regularizers.l2(7e-05), bias_regularizer=tf.keras.regularizers.l2(7e-05), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01), bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01), activation="relu")(input)
    x = layers.BatchNormalization()(x)
    return x
    
def generateFC(shape, input):
    x = layers.Dense(shape, kernel_regularizer=tf.keras.regularizers.l2(7e-05), bias_regularizer=tf.keras.regularizers.l2(7e-05), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01), bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01), activation='relu')(input)
    x = layers.BatchNormalization()(x)
    return x

def getImageModel():

    x = layers.Conv2D(64, (6,2),activation="relu", kernel_regularizer=tf.keras.regularizers.l2(7e-05), bias_regularizer=tf.keras.regularizers.l2(7e-05), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01), bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01), input_shape=imgInputShape)(inputsImgModel)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((3,3))(x)
    for i in range(6):
        print(i)
        x = generateConvolutionLayer(64, (5,1), x)
    outputImg = layers.MaxPool2D((3,3))(x)
    return outputImg



def getActionStateModel():
    
     x = generateFC(256,inputsActionState)
     #x  = layers.add([x])
     x = generateFC(64, x)
     output_state_action = layers.Reshape((1, 1, 64))(x)
     return output_state_action

def generateModel():
    inputImg = getImageModel()
    print("img")
    actionState = getActionStateModel()
    print("action-state")
    x = layers.add([inputImg, actionState])
    for _ in range(6):
        x =  generateConvolutionLayer(64, (3,1),x)
    x = layers.MaxPool2D((2,2))(x)
    for _ in range(3):
        x = generateConvolutionLayer(64, (3,1),x)
    x = layers.Dense(64, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(7e-05), bias_regularizer=tf.keras.regularizers.l2(7e-05), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01), bias_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)    
    return x


outputs = generateModel()
model = keras.Model(inputs=[inputsImgModel, inputsActionState], outputs = outputs)


model.compile(loss=tf.keras.losses.Loss(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
              metrics=['accuracy'])
keras.utils.plot_model(model, to_file="./test.png", show_shapes=True, show_layer_names=True)
model.summary()

        
    



