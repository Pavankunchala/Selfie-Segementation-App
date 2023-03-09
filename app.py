import cv2
import mediapipe as mp

import tempfile
import streamlit as st
import numpy as np

from PIL import Image



#demo video 
DEMO_VIDEO = 'ri1.mp4'


BG_IMAGE = '21.jpg'

BG_COLOR = (192, 192, 192)


mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation





def main():

    #title 
    st.title('Selfie Segementation App')

    #sidebar title
    st.sidebar.title('Segementation App')

    st.sidebar.subheader('Parameters')
    #creating a button for webcam
    use_webcam = st.sidebar.button('Use Webcam')
    #creating a slider for detection confidence 
    
    
    #model selection 
    model_selection = st.sidebar.selectbox('Model Selection',options=[0,1])
    st.markdown(' ## Output')
    stframe = st.empty()
    #background image 
   

    
    #file uploader
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:

        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    #background file buffer
    img_file_buffer = st.sidebar.file_uploader("Upload the background image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = BG_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    #values 
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))


    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)


    with mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection) as selfie_segmentation:
        bg_image = image

        bg_image = cv2.resize(bg_image, (width, height))
        
        while vid.isOpened():

            ret, frame = vid.read()

            if not ret:
                break
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = selfie_segmentation.process(frame)
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

            

            if bg_image is None:
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR

            output_image = np.where(condition, frame, bg_image)

            stframe.image(output_image,use_column_width=True)

        vid.release()
        out.release()
        cv2.destroyAllWindows()

    st.success('Video is Processed')
    st.stop()

if __name__ == '__main__':
    main()
        





