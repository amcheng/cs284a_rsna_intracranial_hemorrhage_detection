from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers import Conv2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D
from keras.layers import concatenate, multiply, add

# converted SE-ResNeXt50 layers from Caffe:
# https://github.com/hujie-frank/SENet/blob/master/models/SE-ResNeXt-50.prototxt

# adapted specific InceptionV3 layers from:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py

class CustomModel:
    def __init__(self):
        self.model_name = "CustomModel"
        self.num_classes = 6
        self.input_dims = (224,224,3)
        self.batch_size = 16
        self.num_epochs = 5
        self.learning_rate = 5e-4
        self.decay_rate = 0.8
        self.decay_steps = 1
        self.verbose = 1
        self._build()
   
    def _build(self):
        x = Input(shape=self.input_dims)
       
        # basic conv,bn,relu layer
        conv1 = self.build_conv_layer(x, 64,kernel_size=(7,7),strides=2,padding='same')
        # output size = 112x112x64
       
        conv1 = self.build_conv_layer(conv1, 256,kernel_size=(1,1),strides=1,padding='same')
        # output size = 112x112x256
       
        # inception module - pick up generic features at different scales
        inception1 = self.build_inception_block1(conv1, num_repeat=1)
        # output size = 112x112x256
       
        reshape1 = MaxPooling2D(pool_size=(3,3),strides=2)(inception1)
        # output size = 56x56x256
   
        # seresnext module - pass forward relevant info and get interesting features
        seresnext1 = self.build_seresnext_block(reshape1, num_repeat=3, out_channels=128, fc=[16, 256])
        # output size = 56x56x256
       
        reshape2 = MaxPooling2D(pool_size=(3,3),strides=2)(seresnext1)
        reshape2 = self.build_conv_layer(reshape2,512,kernel_size=1,strides=1,padding='same')
        # output size = 28x28x512
       
        seresnext2 = self.build_seresnext_block(reshape2, num_repeat=4, out_channels=256, fc=[32, 512])
        # output size = 28x28x512
       
        reshape3 = MaxPooling2D(pool_size=(3,3),strides=2)(seresnext2)
        reshape3 = self.build_conv_layer(reshape3,1024,kernel_size=6,strides=1,padding='same')
        # output size = 14x14x1024
       
        seresnext3 = self.build_seresnext_block(reshape3, num_repeat=4, out_channels=512, fc=[64, 1024])
        # output size = 14x14x1024
       
        reshape4 = MaxPooling2D(pool_size=(3,3),strides=2)(seresnext3)
        reshape4 = self.build_conv_layer(reshape4,2048,kernel_size=1,strides=1,padding='same')
        # output size = 7x7x2048
       
        seresnext4 = self.build_seresnext_block(reshape4, num_repeat=3, out_channels=1024, fc=[128, 2048])
        # output size = 7x7x2048
       
        inception2 = self.build_inception_block2(reshape4, num_repeat=1)
        # output size = 7x7x2048
   
        # wrap-up fc,softmax layer
        out = GlobalAveragePooling2D()(seresnext4)
        out = Dense(self.num_classes, activation="softmax")(out)
       
        self.model = Model(x, out)
        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])
   
    def build_inception_block1(self, x, num_repeat):
        # input and output shapes are the same, to make things easier
        for i in range(num_repeat):
            branch1x1 = self.build_conv_layer(x, 64, kernel_size=(1,1),padding='same', strides=(1,1))

            branch5x5 = self.build_conv_layer(x, 48, kernel_size=(1,1),padding='same', strides=(1,1))
            branch5x5 = self.build_conv_layer(branch5x5, 64, kernel_size=(1,1),padding='same', strides=(1,1))

            branch3x3dbl = self.build_conv_layer(x, 64, kernel_size=(1,1),padding='same', strides=(1,1))
            branch3x3dbl = self.build_conv_layer(branch3x3dbl, 96, kernel_size=(3,3),padding='same', strides=(1,1))
            branch3x3dbl = self.build_conv_layer(branch3x3dbl, 96, kernel_size=(3,3),padding='same', strides=(1,1))

            branch_pool = AveragePooling2D(pool_size=(3, 3),strides=(1, 1),padding='same')(x)
            branch_pool = self.build_conv_layer(branch_pool, 32, kernel_size=(1,1),padding='same', strides=(1,1))
           
            x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],axis=3)
       
        return x
   
    def build_inception_block2(self, x, num_repeat):
        for i in range(num_repeat):
            branch1x1 = self.build_conv_layer(x, 320, kernel_size=(1,1),padding='same', strides=(1,1))

            branch3x3 = self.build_conv_layer(x, 384, kernel_size=(1,1),padding='same', strides=(1,1))
            branch3x3_1 = self.build_conv_layer(branch3x3, 384, kernel_size=(1,3),padding='same', strides=(1,1))
            branch3x3_2 = self.build_conv_layer(branch3x3, 384, kernel_size=(3,1),padding='same', strides=(1,1))
            branch3x3 = concatenate([branch3x3_1, branch3x3_2],axis=3)

            branch3x3dbl = self.build_conv_layer(x, 448, kernel_size=(1,1),padding='same', strides=(1,1))
            branch3x3dbl = self.build_conv_layer(branch3x3dbl, 384, kernel_size=(3,3),padding='same', strides=(1,1))
            branch3x3dbl_1 = self.build_conv_layer(branch3x3dbl, 384, kernel_size=(1,3),padding='same', strides=(1,1))
            branch3x3dbl_2 = self.build_conv_layer(branch3x3dbl, 384, kernel_size=(3,1),padding='same', strides=(1,1))
            branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2],axis=3)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.build_conv_layer(branch_pool, 192, kernel_size=(1,1),padding='same', strides=(1,1))
            x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],axis=3)
        return x
   
    def build_seresnext_block(self, x, num_repeat, out_channels, fc):
        # input and output shapes of a module (minus fc layers) are the same.
        for i in range(num_repeat):
            # build a layer
            residual = x
            curr = self.build_current_layer(x, out_channels)
            se = self.build_se_layer(curr, fc) # already includes scale layer
            x = add([residual, se])
            x = Activation('relu')(x)
        return x
   
    def build_conv_layer(self, x, out_channels, kernel_size, padding, strides, add_relu=True):
        x = Conv2D(filters=out_channels,kernel_size=kernel_size,padding=padding,strides=strides)(x)
        x = BatchNormalization()(x)
        if add_relu:
            x = Activation('relu')(x)
        return x

    def build_mid_layer(self, x, out_channels, kernel_size, cardinality):
        branches = list()
        for i in range(cardinality):
            branch = self.build_conv_layer(x, out_channels//cardinality, kernel_size, padding='same', strides=(1,1))
            branches.append(branch)
        x = concatenate(branches, axis=3)
        return x

    def build_current_layer(self, x, out_channels):
        x = self.build_conv_layer(x, out_channels, kernel_size=1, padding='same', strides=(1,1))
        x = self.build_mid_layer(x, out_channels, kernel_size=3, cardinality=32)
        x = self.build_conv_layer(x, out_channels*2, kernel_size=1, padding='same', strides=(1,1),add_relu=False)
        return x

    def build_se_layer(self, x, fc):
        squeeze = GlobalAveragePooling2D()(x)
        excitation = Dense(fc[0])(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(fc[1])(excitation)
        excitation = Activation('sigmoid')(excitation)
        scale = multiply([x,excitation])
        return scale
   
    # factory functions
   
    def fit_and_predict(self, train_df, valid_df, test_df):
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.model_name, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
       
        self.model.fit_generator(
            DataGenerator(
                train_df.index,
                train_df,
                self.batch_size,
                self.input_dims,
                train_images_dir
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=4,
#             steps_per_epoch=1500,
            callbacks=[pred_history, scheduler, checkpointer]
        )
        return pred_history
   
    def save(self, path):
        self.model.save_weights(path)
   
    def load(self, path):
        self.model.load_weights(path)
       
    def summary(self):
        print(self.summary())

model = CustomModel()