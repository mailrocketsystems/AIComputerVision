import cv2
import datetime
import imutils


def main():
    cap = cv2.VideoCapture('test_video.mp4')

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if(ret):
            frame = imutils.resize(frame, width=800)
            total_frames = total_frames + 1

            fps_end_time = datetime.datetime.now()
            time_difference = fps_end_time - fps_start_time

            if time_difference.seconds ==0:
                fps=0.0

            else:
                fps = total_frames / time_difference.seconds

            fps_text = "FPS : {:.2f}".format(fps)

            cv2.putText(frame , fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),1)

            cv2.imshow("Application",frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            cv2.destroyAllWindows()
            break

main()
