import cv2 as cv
import os
import asyncio
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL
from PIL import Image

import pandas as pd
import pickle
# from sklearn.svm import LinearSVC
import numpy as np
from DigiOCVIP import Vidcap
import sqlalchemy as sqa
import creds.config
import creds
from creds import *
import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator    #load_img, img_to_array,
from keras.applications.vgg16 import preprocess_input, decode_predictions

# define constants
project_id = 'Slitter2BladePositionClassifier'
camera_name = 'Slitter2BladePositionClassifierCamera'
camip = '10.1.35.44'
caps_folder = project_id+'/'+camera_name

# model_pickle = './resources/Model_151222_IridiumV2.h5''slitter_model_jan20E5Basic.h5'
# model_pickle = './resources/slitter_model_jan20E5Basic.h5'
# model_pickle = './resources/basic_vanadium_sans_preproccing.h5'
model_pickle = './resources/slittercroppedmodel_vanadiumBase_v1.h5'

finalRegion = ((0,305),(512,817))
# define helper functions
# ---------------------------------------------------------------------------------------------------------------------
def make_folder(path):
    try:
        os.mkdir(path)
    except OSError as err:
        print('captures directory ' + caps_folder + ' exists')

def get_img_list(folder):
    return os.listdir(folder)

# class frame_getter:
#     def __init__(self):
#         self.new_frames = []
#         self.new_frames_sobel = []
#         self.getting_active = False
#         self.imgs_df = pd.DataFrame(
#             {
#                 'fileday': [],
#                 'filename': [],
#                 'tags': []
#             }
#         )
# ((457, 170), (535, 311))
# TODO: refine this method so it provides a way to retreive a 512,512 slice of an image, potentially refine this method so you can direct where in the pictoure  to crop at run time
def imcrop(img, bounds):
    return img[bounds[0][0]:bounds[1][0],
           bounds[0][1]:bounds[1][1], :]
#     def get_strip_values(self, img, sobel_k=3, n_stripes=15):
#         imsobel = cv.Sobel(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.CV_8U, 0, 1, ksize=sobel_k)
#         rayscores = []
#         for line in [0]+[math.floor(i * img.shape[1] / (n_stripes - 1)) for i in range(1, n_stripes-1)]+[img.shape[1]-1]:
#             ray = [x for x in imsobel[:, line] if x > 0]
#             # print(ray)
#             if ray:
#                 rayscores += [np.mean(ray) * np.max(ray) / 255]
#             else:
#                 rayscores += [0]
#         return rayscores, imsobel
'''
this method can take a raw image and a set of coordinates.
if the cropped region happens to be too big for whatever reason,
the code automatically resizes the image to be acceptable input for the prediction model
:parameter==
        raw :: cvImage numpy array ::
        approxRegion :: Tuple<Tuple<2><2>> :: {a 2 tuple of 2 coordinates defined at runtime user or by the config file used for region extraction} 
:Output == 
        croppedImg :: cvImage nparray<UInt> :: {unit testing shoudl confirm that this output is always 512,512,3}
'''
from keras.preprocessing.image import ImageDataGenerator
#this is no longer vgg preproccing, might need to add rescaling
def imCrop2(raw):
    h, w, _ = raw.shape
    croppedImg = raw[int(len(raw[0])/5):-1,int(len(raw[0])/2):-1,:]
    ch, cw, cdep = croppedImg.shape
    # preProcced = tf.keras.applications.vgg16.preprocess_input(croppedImg)
    # im = Image.fromarray(np.uint8(preProcced))
    # im.show('cropped')
    im = Image.fromarray(np.uint8(croppedImg))
    # im.show('cropped')
    # if ch != 659 or cw != 959:
    if ch != 695 or cw != 959:
        #execute resize code
        resizedImg = cv.resize(croppedImg,(959, 695), interpolation = cv.INTER_AREA)#I am 90% percent sure this is the right syntax
        # rescaled = resizedImg / 1. / 255
        return resizedImg
    return croppedImg
def save_state(small_door_state):
    #todo pay attention to this BOOKMARK
    engine = sqa.create_engine(
        f'mssql+pymssql://{creds.config.sqluser}:{creds.config.sqlpw}@10.1.16.245',
        echo=False  # set to true to spit out all the SQL code in terminal for debugging
    )
    conn = engine.connect()
    conn.execute('COMMIT')
    conn.execute('EXEC dbo.spZM2_DigiCV_SmallDoorState @doorstate={0};'.format(small_door_state))

    conn.close()
    # conn = pymssql.connect(config.server, config.sqluser, config.sqlpw, config.sqldb)
    # cursor = conn.cursor()
    # res = cursor.execute('EXEC dbo.spZM2_DigiCV_SmallDoorState @doorstate={0};'.format(small_door_state))
    # # why is this stupid
    # print('sql saved')
    # cursor.close()
    # conn.close()


make_folder(project_id)
make_folder(caps_folder)

# define async fucntions
# ---------------------------------------------------------------------------------------------------------------------
# tmr loop frame of intrest finder
def decodePredict(state):
    state = np.argmax(state, axis=-1) # data should be a 1 or 0, [1]
    return state


async def get_frame(vcap, sample_rate):
    classif_mode = True
    if not classif_mode:
        make_folder(caps_folder + '/raw')
    now = pd.Timestamp.now()
    # today_folder = "/"+now.strftime("%Y-%m-%d")
    # make_folder(caps_folder)
    # make_folder(caps_folder+today_folder+xform_folder)

    cntr = 0
    first_in = pd.Timestamp.now()
    # cv.imshow(project_id + '/' + camera_name, vcap.frame)
    last_minute = now.minute
    last_day = now.day
    procs = 0
    if classif_mode:
        #keras load approach
        model = tf.keras.models.load_model(filepath=model_pickle, compile=True,options=None)
        #todo this enum is never used , candidate for refactoring
        states = {
            0: 'doors closed',
            1: 'small-door open',
            2: 'big door open'
        }
    slitter_last_state = 0
    slitter_state_current = 0
    none_cntr = 0
    while 1:
        img = vcap.frame
        if img is not None:
            img_original = img.copy()
            none_cntr = 0
            if classif_mode:
                img = imCrop2(img)
            # Classify image
            if classif_mode:
                print(img.shape)
                img = img.reshape(-1,695, 959,3)
                # img = img.reshape(1,img.shape[0], img.shape[1],img.shape[2])
                state = model.predict(img)
                print(state)
                # state = np.argmax(state, axis=-1)
                #todo reading is fundamental research the lead s on binary regression to make sure of thsi prediciton issue is secure
                if state > 0.5:
                    #maybe its 0.0005
                    #maybe its 0.00038766
                    #0.0004
                    state = 1
                    #active
                else:
                    state = 0
                    #inactive
                slitter_last_state=slitter_state_current
                print(state)
                if state == 0:
                    slitter_state_current = 0
                    #active
                else:
                    slitter_state_current = 1
                    #inactive
                # 1    active +++++++    0     inactive
                # Save state to trending if state change
                if slitter_last_state != slitter_state_current:
                    try:
                        save_state(slitter_state_current)#todo find error reason #'name config is not defined'
                        print('State trended...')
                    except Exception as e:
                        print('Error trending state...')
                        print(e)
                # Save image
                if True:
                    try:
                        cv.imwrite(caps_folder + "/Cap_" + str(cntr) + "_" +
                              str(pd.Timestamp.now()).replace(':', '.').replace(' ', 'T') +
                              '_s'+str(state[0]) +
                              '.png',
                              img_original
                              )
                        print('image saved')
                        cntr += 1  # TODO FAILURE TO SAVE IS BECAUSE CODE IS USING PREPROCESSED VERSION OF THE ORIGINAL IMAGE BE WARY AND CORRECT THIS BUG BY TRACKINGORIGINAL IMAGE STATE
                    except Exception as e:
                        print(e)

                else:
                    print('failed to save image')

            else:
                # Save image
                if cv.imwrite(caps_folder + "/raw/Cap_" + str(cntr) + "_" +
                              str(pd.Timestamp.now()).replace(':', '.').replace(' ', 'T') +
                              # '_s' + str(state[0]) +
                              '.png',
                              img
                              ):
                    print('image saved')
                    cntr += 1
                else:
                    print('failed to save image')

            # cleanup old images when list of files > 10000
            if len(get_img_list(caps_folder)) > 10000:  # TODO: Change back to 10000
                files = get_img_list(caps_folder)
                oldest_file = min([caps_folder + "/" + file for file in files], key=os.path.getctime)
                os.remove(oldest_file)

        else:
            print('none img, check again is persists')
            none_cntr += 1
            if none_cntr == 6:
                break
        await asyncio.sleep(sample_rate)  # check five frames per second, if possible, use a counter to validate procinng rate


# main async runner
async def main():
    # frm_get = frame_getter()
    # frm_get.
    vcap = Vidcap(camip, user='root', pw='root')  # AP4 insp top
    await(
        asyncio.gather(
            get_frame(vcap, 10),  # Extract an image from the realtime feed every n seconds
        )
    )

    return 0


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('exiting service...')
        cv.destroyAllWindows()
        os._exit(0)

'''
                encoding labels are as follows
                if [0,1] then: closed  ::: my key == 0
                if [1,0] then open  ::: my key == 1
                after argmax i get the index of the greates probability
                if index == 1 then key == 0
                if index == 0 then key == 1
'''