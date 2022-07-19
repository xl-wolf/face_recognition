import cv2
import os


def detect(pic_path):
    # read image 读取图片
    img = cv2.imread(pic_path)
    #  convert colorful image to the gray 图片灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # load classifer 加载人脸识别模型
    face_detect = cv2.CascadeClassifier(
        cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    # （灰色转换，每次遍历缩放的倍数，检测几次都成功才检测成功，默认0，人脸最小尺寸，人脸最大尺寸）
    face = face_detect.detectMultiScale(gray, 1.01, 50, 0, (2, 2), (350, 350))

    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)

    filename = pic_path[0:-5]+'after'+pic_path[-5:]
    # save image 保存图片
    cv2.imwrite(filename, img)
    # 用户按下q键退出
    cv2.waitKey(0) == ord('q')


def detect_all_pics():
    for selfPath, dirs, pics in os.walk(r'./facesSrc'):
        for pic in pics:
            if('after' not in pic):
                detect(selfPath+'/'+pic)


if(__name__ == "__main__"):
    detect_all_pics()
