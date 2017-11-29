import os
import cv2
import face as Face


def add_overlays(frame, face, image_rate):
    if face is not None:
        face_bb = face.bounding_box.astype(int)
        cv2.rectangle(frame,
                      (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                      (0, 255, 0), 2)

    cv2.putText(frame, str(image_rate) + "%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                thickness=2, lineType=2)


def main():
    # Number of frames after which to run face detection
    frame_interval = 3
    frame_count = 0
    image_count = 0

    name = raw_input('Enter Your Name: ')
    os.system('mkdir ./train_data/%s' % name)

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    face_capture = Face.Capture()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            face = face_capture.capture(frame)
            if face is not None:
                image_count += 1
                cv2.imwrite('./train_data/%s/%s_%d.jpg' % (name, name, image_count), frame)
        add_overlays(frame, face, image_count*2)

        frame_count += 1
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.resizeWindow('Video', 1280, 720)
        # cv2.moveWindow('Video', 50, 50)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.system('rm -rf ./train_data/%s' % name)
            break

        if image_count == 50:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
