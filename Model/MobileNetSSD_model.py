import keras
import tensorflow as tf

class MobileNetSSD:
    
    def __init__(self, input_shape=(300,300,3), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    
    def build_keras_model(self):
     
        class_num = self.num_classes
        input_shape = self.input_shape
            
        inputs =  tf.keras.layers.Input(shape=input_shape, name="Input")
        # layer 1 : Conv2d + BN 
        # (300,300,3) => (150,150,32)
        x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 2 : DWConv2d + BN 
        # (150,150,32) => (150,150,32)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 3 : PWConv2d + BN 
        # (150,150,32) => (150,150,64)
        x = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 4 : DWConv2d + BN 
        # (150,150,64) => (75,75,64)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 5 : PWConv2d + BN 
        # (75,75,64) => (75,75,128)
        x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 6 : DWConv2d + BN 
        # (75,75,128) => (75,75,128)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 7 : PWConv2d + BN 
        # (75,75,128) => (75,75,128)
        x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 8 : DWConv2d + BN 
        # (75,75,128) => (38,38,128)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 9 : PWConv2d + BN 
        # (38,38,128) => (38,38,256)
        x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 10 : DWConv2d + BN 
        # (38,38,256) => (38,38,256)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 11 : PWConv2d + BN 
        # (38,38,256) => (38,38,256)
        x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 12 : DWConv2d + BN 
        # (38,38,256) => (19,19,256)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 13 : PWConv2d + BN 
        # (19,19,256) => (19,19,512)
        x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # Same 5 times DW + PW : 1
        # layer 14 : DWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 15 : PWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # Same 5 times DW + PW : 2
        # layer 16 : DWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 17 : PWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # Same 5 times DW + PW : 3
        # layer 18 : DWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 19 : PWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # Same 5 times DW + PW : 4
        # layer 20 : DWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 21 : PWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # Same 5 times DW + PW : 5
        # layer 22 : DWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
            
        # layer 23 : PWConv2d + BN 
        # (19,19,512) => (19,19,512)
        x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
    
        
        # feature layer 1 : 
        # (19,19,512) => (19,19, 4 * (class_num | 4)) 
        f1_loc = tf.keras.layers.Conv2D((3 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f1_loc = tf.keras.layers.Reshape(( 19 * 19 * 3, (4)), input_shape=(19,19, 3 * (4)))(f1_loc)
            
        f1_conf = tf.keras.layers.Conv2D((3 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f1_conf = tf.keras.layers.Reshape(( 19 * 19 * 3, (class_num)), input_shape=(19,19, 3 * (class_num)), name = 'f1_conf')(f1_conf)
           
    
        # layer 24,25 : DWConv2d + PWConv2d
        # (19,19,512) => (10,10,512) => (10,10,1024)    
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
         
        x = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
        
    
        # layer 26,27 : DWConv2d + PWConv2d
        # (10,10,1024) => (10,10,1024) => (10,10,1024)    
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
        
        x = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(fused=False)(x)
        x = tf.keras.layers.ReLU(6.)(x)
    
        # layer feature 2 : 
        # (10,10,1024) => (10, 10, 6 * (class_num | 4)) 
        f2_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f2_loc = tf.keras.layers.Reshape(( 10 * 10 * 6, (4)), input_shape=(10, 10, 6 * (4)))(f2_loc)
       
        f2_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f2_conf = tf.keras.layers.Reshape(( 10 * 10 * 6, (class_num)), input_shape=(10, 10, 6 * (class_num)), name = 'f2_conf')(f2_conf)
        
        # layer 28,29 : PWConv2d + Conv2d
        # (10,10,1024) => (10,10,256) => (5,5,512)    
        x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
        
        # layer feature 3 : 
        # (5,5,512) => (5, 5, 6 * (class_num | 4)) 
        f3_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f3_loc = tf.keras.layers.Reshape(( 5 * 5 * 6, (4)), input_shape=(5, 5, 6 * (4)) )(f3_loc)
        
        f3_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f3_conf = tf.keras.layers.Reshape(( 5 * 5 * 6, (class_num)), input_shape=(5, 5, 6 * (class_num)), name = 'f3_conf' )(f3_conf)
        
        # layer 30,31 : PWConv2d + Conv2d
        # (5,5,512) => (5,5,128) => (3,3,256)    
        x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
        
        
        # layer feature 4 : 
        # (3,3,256) => (3, 3, 4 * (class_num | 4)) 
        f4_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f4_loc = tf.keras.layers.Reshape(( 3 * 3 * 6, (4)),  input_shape=(3, 3, 6 * (4)) )(f4_loc)
        
        f4_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f4_conf = tf.keras.layers.Reshape(( 3 * 3 * 6, (class_num)),  input_shape=(3, 3, 6 * (class_num)), name = 'f4_conf' )(f4_conf)
        
        # layer 32,33 : PWConv2d + Conv2d
        # (3,3,256) => (3,3,128) => (2,2,256)    
        x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
        
        # layer feature 5 : 
        # (2,2,256) => (2, 2, 4 * (class_num | 4)) 
        f5_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
        f5_loc = tf.keras.layers.Reshape(( 2 * 2 * 6, (4)),  input_shape=(1, 1, 6 * (4)) )(f5_loc)
        
        f5_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
        f5_conf = tf.keras.layers.Reshape(( 2 * 2 * 6, (class_num)),  input_shape=(1, 1, 6 * (class_num)), name = 'f5_conf' )(f5_conf)
        
        # layer 34,35 : PWConv2d + Conv2d
        # (2,2,256) => (2,2,128) => (1,1,256)    
        x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
        
    
        # layer feature 6 : 
        # (1,1,256) => (1, 1, 4 * (class_num | 4)) 
        f6_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
        f6_loc = tf.keras.layers.Reshape(( 1 * 1 * 6, (4)),  input_shape=(1, 1, 6 * (4)) )(f6_loc)
        
        f6_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
        f6_conf = tf.keras.layers.Reshape(( 1 * 1 * 6, (class_num)),  input_shape=(1, 1, 6 * (class_num)) , name = 'f6_conf')(f6_conf)
    
        z_loc = tf.keras.layers.concatenate([f1_loc, f2_loc, f3_loc, f4_loc, f5_loc, f6_loc], axis=1, name='total_loc')
    
        z_conf = tf.keras.layers.concatenate([f1_conf, f2_conf, f3_conf, f4_conf, f5_conf, f6_conf], axis=1, name='total_conf')
        z_conf_logistic = tf.keras.layers.Activation('softmax')(z_conf)
    
        z_concat = tf.keras.layers.concatenate([z_conf_logistic, z_loc], name='outputs')
    
    
    
    
    
    
        
        ### for SSD 5 features contract
            
      
        # return tf.keras.Model(inputs = inputs, outputs = [z_loc, z_conf])
    
        return tf.keras.Model(inputs = inputs, outputs = z_concat)