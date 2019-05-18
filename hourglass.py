def res_block(cnet, x):

    cnet = tf.layers.conv2d(cnet, 128, 1, strides=1, padding='same')
    cnet = tf.layers.batch_normalization(cnet)
    cnet = tf.nn.relu(cnet)
    
    cnet = tf.layers.conv2d(cnet, 128, 3, strides=1, padding='same')
    cnet = tf.layers.batch_normalization(cnet)
    cnet = tf.nn.relu(cnet)
    
    cnet = tf.layers.conv2d(cnet, 256, 1, strides=1, padding='same')
    cnet = tf.add(cnet, x)
    x = tf.identity(cnet)
    cnet = tf.layers.batch_normalization(cnet)
    cnet = tf.nn.relu(cnet)

    return cnet, x

def network(img, reuse=False):
    # network variables
    with tf.variable_scope('network', reuse=reuse):
        
        # cnet = tf.nn.atrous_conv2d(cnet, 256, 2, padding='same')
        # cnet = tf.reshape(img, [-1, nimg_h, nimg_w, 1])
        
        # 256
        cnet = tf.layers.conv2d(img, 64, 7, strides=2, padding='same')
        cnet = tf.layers.batch_normalization(cnet)
        cnet = tf.nn.relu(cnet)

        # 128
        cnet = tf.layers.conv2d(cnet, 256, 3, strides=1, padding='same')
        x = tf.identity(cnet)
        cnet = tf.layers.batch_normalization(cnet)
        cnet = tf.nn.relu(cnet)

        cnet, x = res_block(cnet, x)
        x = tf.layers.max_pooling2d(x, 2, 2)
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)

        # 64
        cnet, x = res_block(cnet, x)
        x = tf.layers.max_pooling2d(x, 2, 2)
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)

        # 32
        cnet, x = res_block(cnet, x)

        x32 = tf.identity(x)
        cnet32, x32 = res_block(cnet, x32)

        x = tf.layers.max_pooling2d(x, 2, 2)
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)

        # 16
        cnet, x = res_block(cnet, x)

        x16 = tf.identity(x)
        cnet16, x16 = res_block(cnet, x16)

        x = tf.layers.max_pooling2d(x, 2, 2)
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)

        # 8
        cnet, x = res_block(cnet, x)

        x8 = tf.identity(x)
        cnet8, x8 = res_block(cnet, x8)

        x = tf.layers.max_pooling2d(x, 2, 2)
        cnet = tf.layers.max_pooling2d(cnet, 2, 2)

        # 4
        cnet, x = res_block(cnet, x)

        x = tf.image.resize_images(x, (8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cnet = tf.image.resize_images(cnet, (8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 8
        cnet, x = res_block(cnet, x)
        cnet = tf.add(cnet, cnet8)

        x = tf.image.resize_images(x, (16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cnet = tf.image.resize_images(cnet, (16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 16
        cnet, x = res_block(cnet, x)
        cnet = tf.add(cnet, cnet16)

        x = tf.image.resize_images(x, (32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cnet = tf.image.resize_images(cnet, (32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 32
        cnet, x = res_block(cnet, x)
        cnet = tf.add(cnet, cnet32)

        x = tf.image.resize_images(x, (64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cnet = tf.image.resize_images(cnet, (64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 64
        cnet, x = res_block(cnet, x)

        x = tf.image.resize_images(x, (128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cnet = tf.image.resize_images(cnet, (128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # 128
        cnet = tf.layers.conv2d(cnet, 64, 1, strides=1, padding='same')
        cnet = tf.layers.batch_normalization(cnet)
        cnet = tf.nn.relu(cnet)
        
        cnet = tf.layers.conv2d(cnet, 9, 1, strides=1, padding='same')
        cnet = tf.layers.batch_normalization(cnet)
        cnet = tf.nn.relu(cnet)

        return cnet