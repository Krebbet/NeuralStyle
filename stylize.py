# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import vgg

import tensorflow as tf
import numpy as np

from sys import stderr

import os
from PIL import Image

# This defines the layers that are used in the loss functions Lc, Ls!
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


#for tensor board
#LOGDIR = 'C:\TensorFlow\neural-style-master\neural-style-master\logs'
#LOGDIR = 'C:\datasets\mnist_tut'
#LOGDIR = 'C:\TensorFlow\neural-style-master\neural-style-master\logs'
LOGDIR = './logs'

try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None,
        rContent=False,rStyle=False,label='label'):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    
    print(rContent,rStyle)
    print('Stylize Begin')
    
    # creates the shapes for the 'content' image and the array of 'style' images
    # the initial (1,) is for??
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    
    print('Load VGG Network')
    # load the vgg image classification network....
    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    # apply user defined weights to cutomize style response.
    # through the user settings you can decrease layer weight exponentially with a decay coef.
    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    # so sum(layer_weights} = 1
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum
    
    print('Compute Content')
    #if a style set to zero.
    if rStyle == False:
      # compute content features in feedforward mode
      # This is effectively a constant during processing...    
      g = tf.Graph()
      with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
          image = tf.placeholder('float', shape=shape)
          net = vgg.net_preloaded(vgg_weights, image, pooling)
          #preprocess the input image
          content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
          # calculate the 'content' in each conv layer.
          # Q: this seems inefficient since it calcs each layer seperately..
          #  is there a way to do one calc and access all the values???
          for layer in CONTENT_LAYERS:
              content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    print('Compute Style')
    if rContent == False:

      # compute style features in feedforward mode
      # this again will be a constant in the routine...
      for i in range(len(styles)):
          g = tf.Graph()
          with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
              #set input placeholder shape
              image = tf.placeholder('float', shape=style_shapes[i])
              # load the vgg conv portion of the net
              net = vgg.net_preloaded(vgg_weights, image, pooling)
              # preprocess the input image
              style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
              for layer in STYLE_LAYERS:
                  # feed in the image to each layer of the net
                  # note: this might be where the thing is slow... are we calculating the first layer 5 times???
                  features = net[layer].eval(feed_dict={image: style_pre})
                  features = np.reshape(features, (-1, features.shape[3]))# what is this for???
                  # calculate the gram.
                  gram = np.matmul(features.T, features) / features.size
                  style_features[i][layer] = gram

                
    # We now have the content tensor and the style tensor for our loss functions...
    # We can now 'train' our image...
                
                
    initial_content_noise_coeff = 1.0 - initial_noiseblend
    # make stylized image using backpropogation
    with tf.Graph().as_default():
        # we define our starting image (x)
        # try using the content image here and see
        # how the style progresses..
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)# this line is useless
            initial = tf.random_normal(shape) * 0.256
            # why do we upscale the image by .256???
        else:
            # noise up our input image...
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        
        # Load the net with the inital image as a VARIABLE.
        # This makes it a part of the optimization process!
        image = tf.Variable(initial,name ='X')
        tf.summary.histogram("X", image)
        
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss -> There are wieght applied here that are not described in the paper...
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend
        content_loss = 0
        
        
        with tf.name_scope('Content_Loss'):

          if rStyle == True:
            content_loss = tf.constant(0,dtype = 'float32')
          else:  
            # create calculate content loss operation.
            content_losses = []
            for content_layer in CONTENT_LAYERS:
                content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                        net[content_layer] - content_features[content_layer]) /
                        content_features[content_layer].size))
            content_loss += reduce(tf.add, content_losses)

          
          
        with tf.name_scope('Style_Loss'):
          if rContent == True:
            style_loss = tf.constant(0,dtype = 'float32')
          else:
            # create style loss operation
            # style loss
            style_loss = 0
            for i in range(len(styles)):
                style_losses = []
                for style_layer in STYLE_LAYERS:
                    layer = net[style_layer]
                    _, height, width, number = map(lambda i: i.value, layer.get_shape())
                    size = height * width * number
                    feats = tf.reshape(layer, (-1, number))
                    gram = tf.matmul(tf.transpose(feats), feats) / size
                    style_gram = style_features[i][style_layer]
                    style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
                style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
        
        
        with tf.name_scope('TVD_Loss'):
        
          # total variation denoising ?????
          # What is this denoising processs??? -> regularization of sorts?
          tv_y_size = _tensor_size(image[:,1:,:,:])
          tv_x_size = _tensor_size(image[:,:,1:,:])
          tv_loss = tv_weight * 2 * (
                  (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                      tv_y_size) +
                  (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                      tv_x_size))
          # create overall loss operation...
        with tf.name_scope('Loss'):
          loss = content_loss + style_loss + tv_loss
        
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('TV_Loss', tv_loss)
        tf.summary.scalar('Style_Loss', style_loss)
        tf.summary.scalar('Content_Loss', content_loss)
        summ = tf.summary.merge_all()
        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)
        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())
      
        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(os.path.join(LOGDIR , label))
            writer.add_graph(sess.graph)            
            
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                # run a train step...
                
                #if i % 1 == 0:
                  #[s] = sess.run([summ])
                  #writer.add_summary(s)  
                _,s = sess.run([train_step,summ])  
                #[s] = train_step.run([summ])
                writer.add_summary(s)                    
            
                
                #output anything needed for user....
                
                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    # calculate loss
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        # clac best image...
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))


                    #lName = '%s%d' % (label ,i)
                    #im_sum = tf.summary.image(lName, img_out, 1)
                    #writer.add_summary(im_sum)
                    yield (
                        (None if last_step else i),
                        img_out
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb
