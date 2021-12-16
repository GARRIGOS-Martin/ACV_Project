#!/usr/bin/env python
# coding: utf-8

# In[1]:

import cv2
import numpy
import math
import mediapipe as mp
import numpy as np

# In[2]:

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# In[3]:

def get_face_landmarks(img):
                      
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert BGR to RGB
    results = face_mesh.process(img)
    
    return results

# In[4]:

def draw_face_landmarks(img):
    
    results = get_face_landmarks(image)
    new_img = img.copy()
    
    if results.multi_face_landmarks : #condition on the existence of results for the face landmarks
        for face_landmarks in results.multi_face_landmarks :
            mp_drawing.draw_landmarks(image=new_img,
                                      landmark_list = face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    return new_img

# In[5]:

#---------------------------------------------------------------------------------------------------------------------------
#                                                   SIMPLE FILTERS
#---------------------------------------------------------------------------------------------------------------------------
def sharpening(img):
    
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float)
    img_sharpen = cv2.filter2D(img,-1,kernel)
    
    return img_sharpen

def black_and_white(image):
    return_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return_image[:, :, 0] = gray
    return_image[:, :, 1] = gray
    return_image[:, :, 2] = gray

    return return_image

def sidebyside(imageleft,imageright):
    # get shape of each image
    hl, wl = imageleft.shape[:2]
    hr, wr = imageright.shape[:2]
    # define shape of final image and create it
    newimheight = np.maximum(hl,hr)
    newimwidth = wl+wr
    newim = np.ones([newimheight,newimwidth,3], np.uint8)*255
    # insert images into final image
    newim[:hl,:wl,:] = imageleft.copy()
    newim[:hr,wl:,:] = imageright.copy()

    return newim

def mirror_l(image):

    hr, wr = image.shape[:2]
    w2 = np.int(np.round(wr/2))
    
    image_l = image[:,:w2,:]
    image_r = image_l[:,::-1,:]
    
    image_mir = sidebyside(image_l,image_r)

    return image_mir

def mirror_r(image):

    hr, wr = image.shape[:2]
    w2 = np.int(np.round(wr/2))

    image_r = image[:,w2::,:]
    image_l = image_r[:,::-1,:]

    image_mir = sidebyside(image_l,image_r)

    return image_mir

def to_sepia(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32)/255
    #solid color
    sepia = np.ones(image.shape)
    sepia[:,:,0] *= 153 #B
    sepia[:,:,1] *= 204 #G
    sepia[:,:,2] *= 255 #R
    #hadamard
    sepia[:,:,0] *= normalized_gray #R
    sepia[:,:,1] *= normalized_gray #G
    sepia[:,:,2] *= normalized_gray #B
    return np.array(sepia, np.uint8)

#---------------------------------------------------------------------------------------------------------------------------
#                                                  IMAGE SEGMENTATION
#---------------------------------------------------------------------------------------------------------------------------

mp_selfie_segmentation = mp.solutions.selfie_segmentation

#---------------------------------------------------------------------------------------------------------------------------
#                                                  IMAGE BLENDING FILTERS
#---------------------------------------------------------------------------------------------------------------------------

# In[6]:

def compute_angle(point1,point2):
    
    x1, y1, x2, y2 = point1[0],point1[1],point2[0],point2[1]
    angle = -180/math.pi*math.atan(float(y2-y1)/float(x2-x1))
    
    return angle

# In[7]:

def blend_img_with_overlay(img, overlay_img, blending_pos_x,blending_pos_y):
    
    img_h, img_w = img.shape[:2]
    over_h, over_w = overlay_img.shape[:2]
    
    crop_left = 0
    crop_right = 0
    if blending_pos_y<0:
        crop_left = -blending_pos_y
    elif blending_pos_y + over_w > img_w :
        crop_right = blending_pos_y+over_w+img_w
        
    crop_top = 0
    crop_bottom = 0
    if blending_pos_x<0:
        crop_bottom = -blending_pos_x
    elif blending_pos_x + over_h > img_h :
        crop_top = blending_pos_x+over_h+img_h
    
    new_img = img.copy()
    
    pos_x2 = blending_pos_x + over_h
    pos_y2 = blending_pos_y + over_w
    
    # overlayMask = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    # _, overlayMask = cv2.cvtColor(overlayMask, 10, 1, cv2.THRESH_BINARY_INV)
    
    extOverlay = np.zeros(img.shape, np.uint8)
    extOverlay[blending_pos_x+crop_bottom:pos_x2-crop_top, blending_pos_y+crop_left:pos_y2-crop_right] = overlay_img[crop_bottom:over_h-crop_top,crop_left:over_w-crop_right,:3]
    
    new_img[extOverlay>0] = extOverlay[extOverlay>0]
    
    return new_img


def lens_filter(img, png_fname):

    results = get_face_landmarks(img)
    doggy_ears = cv2.imread(png_fname, cv2.IMREAD_UNCHANGED) # read the image with opencv in another window than img
    # print(doggy_ears.shape) # on vérifie qu'il y a 4 channels (le dernier pour alpha) pour vérifier que c'est bien un png
    # # si c'est pas le cas et qu'on a que 3 channels, on met le param IMREAD UNCHANGED pour que opencv change pas les channels en lisant
    # # l'image
    new_img = img.copy()

    if results.multi_face_landmarks:
        # on veut l'index de 2 points sur le crane à gauche et à droite où seraient les oreilles de l'img png
        # on ouvre la carte des landmark du visage et on regarde : 332, 103
        face_landmarks = results.multi_face_landmarks[0].landmark # pour le 1er visage

        dog_h, dog_w = doggy_ears.shape[:2] # on récup les dimensions de l'img pour plus bas
        face_pin_1 = face_landmarks[332]
        face_pin_2 = face_landmarks[103]

        # on calcule l'angle entre ces deux pts grâce à une fonction définit plus haut
        angle = compute_angle((face_pin_1.x, face_pin_1.y), (face_pin_2.x, face_pin_2.y))

        # on veut rotationner l'img en fonction de l'angle calculé qui correspond à l'angle du visage
        # voir le notebook day2 ACV
        M = cv2.getRotationMatrix2D((dog_w/2, dog_h/2), angle, 1) # on compute la matrix pour faire la rotation
        # get cos and sin from the tansformation matrix T
        cos = np.abs((M[0,0]))
        sin = np.abs((M[0,1]))
        
        #compute new dimensions depending on cos and sin
        new_dog_w = int(dog_h * sin + dog_w * cos)
        new_dog_h = int(dog_w * sin + dog_h * cos)
        
        # Shift transformation matrix
        M[0,2] += int(new_dog_w/2 - dog_w/2)
        M[1,2] += int(new_dog_h/2 - dog_h/2)
        
        doggy_ears = cv2.warpAffine(doggy_ears, M, (new_dog_w, new_dog_h))

        # resize image of doggy_ears for them to match the scale of face
        # on va regarder les points landmarks du visage à utiliser pour avoir l'échelle du visage
        face_right = face_landmarks[454] # pts le plus à droite du visage
        face_left = face_landmarks[234] # pts le plus à gauche
        
        face_top = face_landmarks[10] # pts le plus haut du visage
        face_bottom = face_landmarks[152] # pts le plus bas du visage

        # on calcule la largeur du visage
        face_w = math.sqrt((face_right.x - face_left.x)**2 + (face_right.y - face_left.y)**2)
        # on calcule la longueur du visage
        face_h = math.sqrt((face_top.x - face_bottom.x)**2 + (face_top.y - face_bottom.y)**2)

        # on veut changer les dimensions des doggy ears avec un ratio pour la largeur et un pour la hauteur
        img_h, img_w = img.shape[:2] # dimensions de l'img de base affichée sur la caméra

        ratio_w = (face_w * img_w) / dog_w
        ratio_h = (img_h * face_h) / dog_h

        # on resize les doggy ears pour qu'elles soient aux même dimensions que le visage
        doggy_ears = cv2.resize(doggy_ears, # img à resize
                    (int(ratio_w * dog_w), int(dog_h*ratio_w))) # nvelles dimensions de l'img

        # on veut blend l'img avec les doggy ears. on cherche la position des ears sur l'image
        # /!\ dans opencv, x et y sont inversés mais pas dans mediapipe
        dog_h, dog_w = doggy_ears.shape[:2] # les dim ont changé vu qu'on resize, on récup les nvelles valeurs

        pos_x = int(img_h * face_top.y - dog_h/2)
        pos_y = int(img_w * face_top.x - dog_w/2)

        # on utile une fonction pour blend qu'on a définit plus haut
        new_img = blend_img_with_overlay(img, doggy_ears, pos_x, pos_y)

        # on veut juste récup les oreilles et pas toute l'img, pour pas avoir le fond noir.
        # pour ca on va créer un masque qui est une matrice de booleans. Voir fonction pour blend
        
    return new_img

#---------------------------------------------------------------------------------------------------------------------------
#                                                  ADD EYES IF SLEEPING
#---------------------------------------------------------------------------------------------------------------------------

def ratio_close_eyes(face_landmarks):
    face_place_1 = face_landmarks[159].y
    face_place_2 = face_landmarks[27].y
    face_place_3 = face_landmarks[145].y
    face_place_4 = face_landmarks[386].y
    face_place_5 = face_landmarks[257].y
    face_place_6 = face_landmarks[374].y
    r1 = abs(face_place_1 - face_place_2)
    r2 = abs(face_place_1 - face_place_3)
    r3 = abs(face_place_4 - face_place_5)
    r4 = abs(face_place_4 - face_place_6)
    if r1 < r2 and r3 < r4 :
        result = 1
    else :
        result = 0
    return result

def eyes_filter(img, png_fname):
    results = get_face_landmarks(img)
    big_eyes = cv2.imread(png_fname, cv2.IMREAD_UNCHANGED)
    new_img = img.copy()
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        if ratio_close_eyes(face_landmarks) == 0 :
            eyes_h, eyes_w = big_eyes.shape[:2]
            face_pin_1 = face_landmarks[348]
            face_pin_2 = face_landmarks[119]
            angle = compute_angle((face_pin_1.x, face_pin_1.y), (face_pin_2.x, face_pin_2.y))
            M = cv2.getRotationMatrix2D((eyes_w/2, eyes_h/2), angle, 1)
            big_eyes = cv2.warpAffine(big_eyes,
                            M,
                            (eyes_w, eyes_h))
            face_right = face_landmarks[454]
            face_left = face_landmarks[234]
            face_top = face_landmarks[10]
            face_bottom = face_landmarks[152]
            face_w = math.sqrt((face_right.x - face_left.x)**2 + (face_right.y - face_left.y)**2)
            face_h = math.sqrt((face_top.x - face_bottom.x)**2 + (face_top.y - face_bottom.y)**2)
            img_h, img_w = img.shape[:2]
            ratio_w = (face_w * img_w) / eyes_w
            ratio_h = (img_h * face_h) / eyes_h
            big_eyes = cv2.resize(big_eyes,
                        (int(ratio_w * eyes_w), int(eyes_h*ratio_h)))
            eyes_h, eyes_w = big_eyes.shape[:2]
            pos_x = int(img_h * face_top.y - eyes_h/2)
            pos_y = int(img_w * face_top.x - eyes_w/2)
            big_eyes = blend_img_with_overlay(img, big_eyes, pos_x, pos_y)
            return big_eyes
    return img

#---------------------------------------------------------------------------------------------------------------------------
#                                                  MOSAIC
#---------------------------------------------------------------------------------------------------------------------------

def Text_center(text, w, font) :
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 2, 2)[0]
    return (w - textsize[0]) / 2
    # cv2.putText(img, text, (textX, textY ), font, 1, (255,

def mosaic(image) :
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    # create the new image
    h, w, dim = image.shape
    newim = np.ones([3*h,3*w, dim], np.uint8)

    h_dec = h-30
    # first line
    newim[:h,:w,:] = to_sepia(image)
    text = 'Sepia'
    cv2.putText(newim[:h,:w,:],text,(((w-cv2.getTextSize(text, font, 2, 2)[0][0]))-5,h_dec), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    newim[:h,w:2*w,:] = black_and_white(image)
    text = 'Black & White'
    cv2.putText(newim[:h,w:2*w,:],text,(((w-cv2.getTextSize(text, font, 2, 2)[0][0]))-5,h_dec), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    newim[:h,2*w:3*w,:] = image.copy()

    # second line
    newim[h:2*h,:w,:] = mirror_l(image)
    text = 'Mirror left'
    cv2.putText(newim[h:2*h,:w,:] ,text,(((w-cv2.getTextSize(text, font, 2, 2)[0][0]))-5,h_dec), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    newim[h:2*h,w:2*w,:] = image.copy()
    text = 'Original'
    cv2.putText(newim[h:2*h,w:2*w,:],text,(((w-cv2.getTextSize(text, font, 2, 2)[0][0]))-5,h_dec), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    newim[h:2*h,2*w:3*w,:] = mirror_r(image)
    text = 'Mirror right'
    cv2.putText(newim[h:2*h,2*w:3*w,:],text,(((w-cv2.getTextSize(text, font, 2, 2)[0][0]))-5,h_dec), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # third line
    png_eye = './images_png/yeux1.png'
    sleepy_eyes = eyes_filter(image, png_eye)
    newim[2*h:3*h,:w,:] = cv2.resize(sleepy_eyes, (w, h))
    text = 'Sleepy eyes'
    cv2.putText(newim[2*h:3*h,:w,:] ,text,(((w-cv2.getTextSize(text, font, 2, 2)[0][0]))-5,h_dec), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    png_dog = './images_png/doggy_ears.png'
    doggy_ears = lens_filter(image, png_dog)
    newim[2*h:3*h,w:2*w,:] = cv2.resize(doggy_ears, (w, h)) # nvelles dimensions de l'img
    text = 'Doggy ears'
    cv2.putText(newim[2*h:3*h,w:2*w,:] ,text,(((w-cv2.getTextSize(text, font, 2, 2)[0][0]))-5,h_dec), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(newim[h:2*h,2*w:3*w,:],'Red nose', (np.int(w*0.3),h_dec), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
    newim[2*h:3*h,2*w:3*w,:] = image.copy()

    return newim

#---------------------------------------------------------------------------------------------------------------------------
#                                                   DISPLAY
#---------------------------------------------------------------------------------------------------------------------------

# In[8]:

cam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, 
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh :
    while cam.isOpened():
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # cv2.imshow('Webacam', frame)
        # cv2.imshow('Sharpen', sharpening(frame))
        # cv2.imshow('Black&White', black_and_white(frame))
        # cv2.imshow('Sepia', to_sepia(frame))
        # cv2.imshow('Side', sidebyside(frame, frame.copy()))
        # cv2.imshow('Mirror left', mirror_l(frame))
        # png_must = './images_png/doggy_ears.png'
        # cv2.imshow('Doggy_Ears', lens_filter(frame, png_must))
        
        cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
        cv2.imshow('Mosaic', mosaic(frame))

        if cv2.waitKey(1) == 27:#esc key to quit
            break

cam.release()
cv2.destroyAllWindows()

