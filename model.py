# Importing require python packages

import os
import sys
import scipy.io
import argparse
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import tensorflow as tf
import cv2
import numpy as np

# Here we are going to code for the triplet loss function

# First we will calculate the loss between the base image and the generated image hidden layer activation

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    

    a_C_unrolled = tf.reshape(a_C,[m,-1])
    a_G_unrolled = tf.reshape(a_G,[m,-1])
    
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)
    
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A , tf.transpose(A))
    
    return GA

# compute cost between style image and generated image hidden layer activation

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S,[n_H * n_W , n_C]))
    a_G = tf.transpose(tf.reshape(a_G,[n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG))) / (2 * n_C * n_W * n_H)**2
    
    return J_style_layer

# Merging different style layers for better result

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

# Function to calculate the average style cost of STYLE_LAYERS

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]

        a_S = sess.run(out)

        a_G = out
        
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style

# Calculating the total cost

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    J = alpha * J_content + beta * J_style
    
    return J


# Getting base and style images

ap = argparse.ArgumentParser()
ap.add_argument("-bi", "--base_image", type=str,
	help="path to input image")
ap.add_argument("-si","--style_image", type = str,
    help = "path to style image")
args = vars(ap.parse_args())


# Resizing the image to 400 x 300

def resizeImages(filename):

    updatedImage = cv2.resize(filename,(400,300))
    return updatedImage


content_image = cv2.imread(args['base_image'])
content_image = resizeImages(content_image)
content_image = reshape_and_normalize_image(content_image)

style_image = cv2.imread(args['style_image'])
style_image = resizeImages(style_image)
style_image = reshape_and_normalize_image(style_image)

# Generating noisy image

generated_image = generate_noise_image(content_image)


# Creating graph and starting tensorflow session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Loading pre-trained model

model = load_vgg_model("pre-trained model/imagenet-vgg-verydeep-19.mat")

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']
a_C = sess.run(out)
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content,J_style,10,40)

J = total_cost(J_content,J_style,10,40)


# define optimizer
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step 
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)

    sess.run(tf.global_variables_initializer())
    
    
    # Run the noisy input image (initial generated image) through the model. Use assign().

    sess.run(model['input'].assign(input_image))
    
    
    for i in range(num_iterations):

        sess.run(train_step)
        
        generated_image = sess.run(model['input'])
        
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


model_nn(sess, generated_image)