import cv2


if __name__ == '__main__':

    video_path = '/home/horang1804/HDD1/dataset/aihub_sign/004.수어영상/1.Training/원천데이터/REAL/WORD/01_real_word_video/01/NIA_SL_WORD1501_REAL01_D.mp4'

    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        cv2.imshow('img',frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()