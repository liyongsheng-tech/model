import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import random
import pathlib
import os 



#tensorflow会打印许多无关的信息，屏蔽信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#找到需要处理的数据集
data_dir = 'E:/VScode/multi-output-classification/dataset'
data_root = pathlib.Path(data_dir)


#提取所有的文件路径，就是把所有的图片放到一起，用列表的形式,但是此时还不是真正列表的形式，而是下面这种形式
#以列表的形式将图片放到一起
# WindowsPath('E:/VScode/multi-output-classification/dataset/black_jeans/00000001.jpeg')
all_image_paths = list(data_root.glob('*/*'))
image_count = len(all_image_paths)



#打乱所有的数据，并且转换成为真正的列表，变为下面这种形式
#E:\VScode\multi-output-classification\dataset\blue_shirt\00000133.jpg
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)



#后面要对数据进行编码，所以先提取一共有多少类
#['black_jeans', 'black_shoes', 'blue_dress', 'blue_jeans', 'blue_shirt', 'red_dress', 'red_shirt']
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())



#由于多输入，将他们分别提取出来(观察label_names的打印形式，以——分开颜色和衣服类型)

#取出所有颜色分类
color_label_names = set(name.split('_')[0] for name in label_names)
#取出所有衣服类型分类
item_label_names = set(name.split('_')[1] for name in label_names)



#对提取的所有分类进行编码
#{'blue': 0, 'red': 1, 'black': 2}
color_label_to_index = dict((name, index) for index,name in enumerate(color_label_names))
item_label_to_index = dict((name, index) for index,name in enumerate(item_label_names))




#对所有图片的标签进行编码
#['black_jeans', 'blue_dress', 'blue_dress']
all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]



#分别对颜色和衣服类型编码（对所有图片）
color_labels = [color_label_to_index[label.split('_')[0]] for label in all_image_labels]
item_labels = [item_label_to_index[label.split('_')[1]] for label in all_image_labels]



#处理数据
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image/255.0  # normalize to [0,1] range
    image = 2*image-1
    return image


#tf.data处理数据
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices((color_labels, item_labels))

#数据一一对应
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


#分割训练集和测试集
test_count = int(image_count*0.2)
train_count = image_count - test_count


#在dataset划分数据
train_data = image_label_ds.skip(test_count)
test_data = image_label_ds.take(test_count)



BATCH_SIZE = 16
train_data = train_data.shuffle(buffer_size=train_count).repeat(-1)#-1表示一直重复
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(buffer_size=AUTOTUNE)

test_data = test_data.batch(BATCH_SIZE)



#建立模型
#使用MobileNetV2卷积层
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), 
                                               include_top=False,
                                               weights='imagenet')

mobile_net.trianable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x=mobile_net(inputs)
x0 = tf.keras.layers.GlobalAveragePooling2D()(x)  #打平数据

#颜色输入
x1 = tf.keras.layers.Dense(1024, activation='relu')(x0)
out_color = tf.keras.layers.Dense(len(color_label_names), activation='softmax',name='out_color')(x1)

#衣服类型输入
x2 = tf.keras.layers.Dense(1024, activation='relu')(x0)
out_item = tf.keras.layers.Dense(len(item_label_names), activation='softmax',name='out_item')(x2)


model = tf.keras.Model(inputs=inputs,outputs=[out_color, out_item])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss={'out_color':'sparse_categorical_crossentropy',
                    'out_item':'sparse_categorical_crossentropy'},
              metrics=['acc'])


train_steps = train_count//BATCH_SIZE
test_steps = test_count//BATCH_SIZE



model.fit(train_data,
          epochs=15,
          steps_per_epoch=train_steps,
          validation_data=test_data,
          validation_steps=test_steps)





