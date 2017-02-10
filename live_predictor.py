from __future__ import print_function, division
################################################################################
#                                                                        IMPORTS
################################################################################
import cv2
from main import tf
from main import np
from main import os
import argparse

# Visualisation Imports
import matplotlib as mpl
mpl.use('Agg')  # Matplotlib in non-interactive mode

# Neural Net Imports
from nnet.graphops import GraphOps
from nnet.sessions import tf_initialize_vars_from_file
from nnet.sessions import minimal_in_session_predictions
from graphs import create_graph, model_a, model_b, model_c, model_d

# Misc Imports
from main import PRINT_WIDTH
from support import print_headers
from support import Timer
from support import limit_string
from file_support import maybe_make_pardir
from support import verbose_print_done, verbose_print
from support import repeat_array
from support import digit_array_to_int


# ##############################################################################
# SETTINGS
# ##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save_stream", type=str, default=None,
                    help="Record the stream and save it as this file")
settings = parser.parse_args()

# ESTABLISH MODEL SETTINGS
settings.conv_dropout = 0.1
settings.fc_dropout = 0.1
settings.image_chanels = 1
settings.image_size = (54, 54)


# PREPARING OUTPUT VIDEO SETTINGS
# taken from: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
fps = 20  # Frame Rate - Adjust this based on your own webcam
outframe_dims = (640, 480)
if settings.save_stream is not None:
    maybe_make_pardir(settings.save_stream)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(settings.save_stream,
                          fourcc=fourcc,
                          fps=fps,
                          frameSize=outframe_dims)


################################################################################
#                                                              GRAPH AND SESSION
################################################################################
model = model_a
checkpoint_dir = "results/A_02/checkpoints/"
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_max.chk")

# CREATE TENSORFLOW GRAPH
graph = create_graph(logit_func=model, settings=settings)

# PREPARE SESSION
print_headers("TENSORFLOW SESSION", border="=", width=PRINT_WIDTH)

with tf.Session(graph=graph) as sess:
    # GET IMPORTANT OPERATIONS AND TENSORS FROM GRAPH
    g = GraphOps(graph, "X", "Y", "BBOX", "is_training", "digit_logits",
                 "bbox_logits")
    
    # INITIALIZE VARIABLES
    verbose = True
    saver = tf.train.Saver(name="saver")
    tf_initialize_vars_from_file(f=checkpoint_file, s=sess, saver=saver, verbose=verbose)
    
    # INITIALIZE THE VIDEO CAPTURE DEVICE
    verbose_print("Capturing webcam video", verbose, end="")
    cap = cv2.VideoCapture(0)
    verbose_print_done(verbose)
    if settings.save_stream is not None:
        print("Saving video as:", limit_string(settings.save_stream, tail=50))
    
    # CONTINUOUSLY CAPTURE A FRAME, MAKE A PREDICTION, AND DISPLAY/SAVE
    while (1):
        # Take each frame from camera
        _, frame = cap.read()
        
        # PREPARE IMAGE
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        im = cv2.resize(im, settings.image_size)      # Scale image
                 
        # PREDICT
        digits, bboxes = minimal_in_session_predictions(s=sess, X=np.array([im]))
        bboxes = bboxes[0, 4:].reshape(5,4) # Separate row for each digit's bbox
        bboxes = bboxes*repeat_array(np.array(outframe_dims), n=2) # scale bbox to img size
        bboxes = bboxes.astype(dtype=np.int32)
        digits = digits[0]
        n_digits = (digits != 10).sum()
        number = digit_array_to_int(digits, null=10) # The predicted number
        # print(number)
        
        # DRAW BOUNDING BOXES AND TEXT
        font = cv2.FONT_HERSHEY_SIMPLEX
        overlay = frame.copy()
        for n in range(n_digits):
            # Draw Bounding box for current digit
            x1, y1, x2,y2 = bboxes[-n-1]
            cv2.rectangle(overlay,
                          pt1=(x1, y1), pt2=(x2, y2),
                          color=(41, 30, 245),
                          thickness=4,
                          lineType=cv2.LINE_AA)

            #  Overlay Text for current digit
            x = int(x1 + (x2 - x1) / 2 - 5)
            y = y2 - 5
            cv2.putText(overlay,
                        text=str(digits[-n - 1]),
                        org=(x, y),
                        fontFace=font,
                        fontScale=2,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)

        alpha = 0.8
        frame = cv2.addWeighted(src1=overlay, alpha=alpha, src2=frame, beta=1-alpha, gamma=0)

        
        # SHOW FRAME
        cv2.imshow('frame', frame)

        # SAVE THE FRAME
        if settings.save_stream is not None:
            out.write(frame)
        
        # QUIT IF ESCAPE BUTTON PRESSED
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            print("CLOSING WEBCAM CAPTURE")
            break

    print("Closing display window")
    cv2.destroyAllWindows()
    
    print("[TERMINATED]")
