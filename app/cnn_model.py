#encoding=utf-8
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.applications import ResNet50, Xception, VGG16, resnet50, xception, vgg16
from keras.preprocessing import image
import numpy as np


class DogCatModel(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dropout(0.5, input_shape=(4608,)))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.load_weights(filepath='./model/weights.best3.xception_resnet50_vgg16.hdf5')
        self.resnet50 = ResNet50(weights='model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)
        self.xception = Xception(weights='model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)
        self.vgg16 = VGG16(weights='model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)

    def predict(self, pic_path):
        return self.model.predict(self._get_comb_extraction(pic_path))[0][0]


    # 图片转向量
    def _path_to_tensor(self, path,resize):
        img = image.load_img(path, target_size=resize)
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    # 把图片提取特征向量
    def _get_comb_extraction(self, img_path):
        xception_pred = self._get_bottleneck_feature(self.xception, xception.preprocess_input, img_path, (299, 299))
        resnet50_pred = self._get_bottleneck_feature(self.resnet50, resnet50.preprocess_input, img_path, (224, 224))
        vgg16_pred = self._get_bottleneck_feature(self.vgg16, vgg16.preprocess_input, img_path, (224, 224))
        return np.concatenate([xception_pred, resnet50_pred, vgg16_pred], axis=1)

    def _get_bottleneck_feature(self, pretrain_model, preprocessing, img_path, img_resize):
        model = Model(pretrain_model.input, GlobalAveragePooling2D()(pretrain_model.output))
        return model.predict(preprocessing(self._path_to_tensor(img_path, img_resize)))


