# 使用VGG16模型
from keras.applications.vgg16 import VGG16
print('Start build VGG16 -------')
# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

# Create own input format
input = Input(input_shape, name = 'image_input')

# Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

# Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dense(1, activation='softmax', name='predictions')(x)

# Create own model
my_model = Model(input=input, output=x)

# In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
# pirnt model
print('\nThis is my vgg16 model for lung sick detect')
my_model.summary()

