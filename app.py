import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import tensorflow as tf
import cv2

st.header("POSTURE DETECTION WITH MOVENET")


connections = [
    (0, 1), (0, 2),       # nose → eyes
    (1, 3), (2, 4),       # eyes → ears

    (5, 7), (7, 9),       # left shoulder → elbow → wrist
    (6, 8), (8, 10),      # right shoulder → elbow → wrist

    (5, 6),               # shoulders

    (5, 11), (6, 12),     # shoulders → hips
    (11, 12),             # hips

    (11, 13), (13, 15),   # left hip → knee → ankle
    (12, 14), (14, 16)    # right hip → knee → ankle
]


# Load MoveNet
model = tf.saved_model.load("movenet_model")
movenet = model.signatures["serving_default"]


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV image
        img = frame.to_ndarray(format="bgr24") #cv2 needs it in arr

        # Get dimensions of original image
        height, width, _ = img.shape

        # -------------------------
        # PREPROCESS FOR MOVENET
        # -------------------------

        input_img = cv2.cvtColor( #Change order from BGR to RGB cause of movenet
            img,
            cv2.COLOR_BGR2RGB
        )

        input_img = tf.image.resize(
            input_img,
            (192, 192)
        )

        input_img = tf.cast(
            input_img,
            tf.int32
        )

        input_img = tf.expand_dims(
            input_img,
            axis=0
        )

        # -------------------------
        # RUN MOVENET
        # -------------------------

        output = movenet(input_img)

        keypoints = output["output_0"].numpy()[0, 0, :, :]

        # -------------------------
        # DRAW CONNECTIONS
        # -------------------------

        for point1, point2 in connections:

            y1, x1, confidence1 = keypoints[point1]
            y2, x2, confidence2 = keypoints[point2]

            if confidence1 > 0.3 and confidence2 > 0.3:

                x1 = int(x1 * width)
                y1 = int(y1 * height)

                x2 = int(x2 * width)
                y2 = int(y2 * height)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

        # -------------------------
        # DRAW KEYPOINTS
        # -------------------------

        for keypoint in keypoints:

            y, x, confidence = keypoint

            if confidence > 0.3:

                x = int(x * width)
                y = int(y * height)

                cv2.circle(
                    img,
                    (x, y),
                    5,
                    (255, 255, 255),
                    -1
                )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)