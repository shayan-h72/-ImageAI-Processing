import numpy as np
import nibabel as nib
from PIL import Image
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


n_class = 5
path = 'C:/VSC_workapace/UnetData/'

def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

def ProcessImage(path_file,mode='None',norm=True):
    dataset = []
    try:
        fns = glob.glob(path_file)
        for fn in fns:
            _, base, ext = split_filename(fn)
            img = nib.load(fn).get_data().astype(np.float32).squeeze()
            if img.ndim != 3:
                print(f'Only 3D data supported. File {base}{ext} has dimension {img.ndim}. Skipping.')
                continue
            
            if mode =='mask':
                for i in range(img.shape[2]):
                    r_img =  img[:,:,i]
                    dataset.append(r_img)
            
            else:
                for i in range(img.shape[2]):
                    I = Image.fromarray(img[:,:,i], mode='F')

                    oldmin = np.min(I)
                    oldmax = np.max(I)
                    oldrange = oldmax-oldmin

                    newmin = 0
                    newmax = 255
                    newrange = newmax-newmin
                    scale =(I-oldmin)/oldrange
                    if norm:
                        normal_img = ((newrange*scale) + newmin)/255.
                    else:
                        normal_img = ((newrange*scale) + newmin)

                    dataset.append(normal_img)


        return np.array(dataset)
    except Exception as e:
        print(e)
        return 1

def LoadData(path):
    path_to_data = os.path.join(path, 'data/*.nii*')
    path_to_mask = os.path.join(path, 'mask/*.nii*')

    unmask = ProcessImage(path_to_data,mode='unmask',norm=False)
    mask = ProcessImage(path_to_mask,mode='mask')

    return unmask,mask


def Encode_label(mask):
    labelencoder = LabelEncoder()
    n, h, w = mask.shape
    mask_reshape = mask.reshape(-1,1)
    mask_reshape_encoded = labelencoder.fit_transform(mask_reshape)
    return_mask_shape = mask_reshape_encoded.reshape((n,h,w))
    class_weights = class_weight.compute_class_weight('balanced',np.unique(mask_reshape_encoded),mask_reshape_encoded)

    return return_mask_shape,class_weights


unmask,mask = LoadData(path)
print('unique:',np.unique(mask))
encoded_mask, class_weights = Encode_label(mask)

unmask = np.expand_dims(unmask[0:10,...],axis=3)
mask = np.expand_dims(encoded_mask[0:10,...],axis=3)

print(unmask.shape)
print(mask.shape)
x_train,x_test,y_train,y_test = train_test_split(unmask,mask,test_size=0.2,random_state=0)

train_mask_cat = to_categorical(y_train,num_classes=n_class)
y_train_cat = train_mask_cat.reshape((y_train.shape[0],y_train.shape[1],y_train.shape[2],n_class))

test_mask_cat = to_categorical(y_test,num_classes=n_class)
y_test_cat = test_mask_cat.reshape((y_test.shape[0],y_test.shape[1],y_test.shape[2],n_class))


vgg_model = VGG16(include_top=False, weights='imagenet')
vgg_config = vgg_model.get_config()
h,w,c = 512,512,1
vgg_config['layers'][0]['config']['batch_input_shape'] = (None,h,w,c)
vgg_updated= Model.from_config(vgg_config)

#print(vgg_updates.summary())

def avg_ws (weights):
    avg_w = np.mean(weights,axis=2).reshape(weights[:,:,-1:,:].shape)
    return (avg_w)

vgg_updated_conf = vgg_updated.get_config()
vgg_updated_l_names = [vgg_updated_conf['layers'][x]['name'] for x in range(len(vgg_updated_conf['layers']))]
first = vgg_updated_l_names[1]

for layer in vgg_model.layers:
    if layer.name in vgg_updated_l_names:
        if layer.get_weights() != []:
            target_l = vgg_updated.get_layer(layer.name)
            
            if layer.name in first:
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                
                wsc = avg_ws(weights)
                
                target_l.set_weights([wsc,biases])
                target_l.trainable = False
                
            else:
                target_l.set_weights(layer.get_weights())
                target_l.trainable = False  

#print(vgg_updated.summary())

def upsample(filters, size, shape, stride=2, apply_dropout=False,apply_batchnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=stride, batch_input_shape=shape,padding='same',
                                    kernel_initializer=initializer))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.1))

    result.add(tf.keras.layers.ReLU())

    return result

def buildUNet():
    inputs = tf.keras.layers.Input(shape=[512,512,1])
    up_stack = [
        upsample(512, 3, (None, 16, 16, 512), apply_dropout=True), 
        upsample(512, 3, (None, 32, 32, 1024), stride=1), 
        upsample(512, 3, (None, 32, 32, 1024), stride=1), 
        upsample(512, 3, (None, 32, 32, 1024), stride=1), 

        upsample(512, 3, (None, 32, 32, 1024), apply_dropout=True), 
        upsample(512, 3, (None, 64, 64, 1024), stride=1), 
        upsample(512, 3, (None, 64, 64, 1024), stride=1), 
        upsample(512, 3, (None, 64, 64, 1024), stride=1), 

        upsample(256, 3, (None, 64, 64, 768), apply_dropout=True), 
        upsample(256, 3, (None, 128, 128, 512), stride=1), 
        upsample(256, 3, (None, 128, 128, 512), stride=1), 
        upsample(256, 3, (None, 128, 128, 512), stride=1), 

        upsample(128, 3, (None, 128, 128, 384)), 
        upsample(128, 3, (None, 256, 256, 256), stride=1), 
        upsample(96, 3, (None, 256, 256, 256), stride=1), 
        upsample(32, 3, (None, 256, 256, 160)),
        upsample(32, 3, (None, 512, 512, 96),stride=1),
        upsample(16, 3, (None, 512, 512, 96), stride=1), # (bs, 512, 512, 16)

    ]

    prev_last = tf.keras.layers.Conv2DTranspose(8, 3,strides=1,padding='same')
    last = tf.keras.layers.Conv2DTranspose(n_class, 3,padding='same',activation='softmax') # (bs, 512, 512, n_class)

    x = inputs

    skips = []
    down_stack = vgg_updated.layers
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = prev_last(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = buildUNet()
model.compile(optimizer='adam',loss= 'categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train_cat,
                    batch_size=2,
                    verbose=1,
                    epochs=3,
                    validation_data=(x_test,y_test_cat),
                    shuffle=False)