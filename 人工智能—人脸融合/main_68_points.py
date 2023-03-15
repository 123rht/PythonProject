# 本文件代码目的：读取图片，对人脸的68个点进行绘制
# 作者：江大白
# 平台网站：www.jiangdabai.com
# 微信公众号1：大白智能
# 微信公众号2：江大白
# 其他：《30天入门人工智能》视频、《国内TOP50大厂&3000篇面经核心秘籍》等等，尽在平台网站查看
# 祝愿：分享的代码和资料，希望对大家入门人工智能有帮助！一路相伴，共同进步！
# 视频教程：本应用涉及很多人脸识别相关的知识点，人脸识别的入门视频，可以查看平台网站的《深入浅出人脸识别基础及项目应用》

### 导入封装好的一些模块，直接使用已经写好的函数，简单高效
import face_recognition  # 人脸识别开源框架，底层基础也是基于dlib
import dlib  # 加载dlib算法模块
import cv2  # 图像处理的opencv库
import argparse  # 参数解析模块

### 人脸特征点定位函数
def points_face(opt):
    # 加载dlib自带的人脸检测模型
    detector = dlib.get_frontal_face_detector()
    # 加载dlib的特征点定位模型
    predictor = dlib.shape_predictor(opt.landmarks_model_path)
    # 使用opencv中的imread函数，读取原始图像
    img = cv2.imread(opt.image_path)
    # 对读取的原始图像，进行人脸检测
    faceRects = detector(img)
    # 图片中可能存在多张人脸，依次遍历输出每张人脸的位置信息,并在特定位置，添加兔子帽子特效
    for box_info in faceRects:
        # 对读取的原始图像，进行人脸特征点定位
        landmarks = predictor(img, box_info)
        for i in range(68):
            # 使用cv2.circle，在每个关键点的位置，绘制一个圆心，其中3表示圆的半径，-1，表示是一个实心圆
            cv2.circle(img, (landmarks.part(i).x, landmarks.part(i).y), 3, (0, 255, 255), -1)
            # 使用cv2.putText，在每个关键点的位置，画上数值，0.5是数字的大小
            cv2.putText(img, str(i), (landmarks.part(i).x-5, landmarks.part(i).y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # 很多时候，图片的长宽太大，屏幕显示不全
    # 为了便于查看图片，可以将图片的大小，缩放到某一尺寸，比如（500,600），即宽500像素，高600像素
    img = cv2.resize(img, (500, 600))
    # 显示图片
    cv2.imshow("image", img)
    # 显示图片停顿的时间，如果是0，则一直显示。如果是100，则显示100ms
    cv2.waitKey(0)

### 函数主入口
if __name__ == '__main__':
    # 新建一个解析器
    parser = argparse.ArgumentParser()
    # 为解析器添加选项，比如为image_path，添加图片地址，可以更换其他图片尝试。（在default后面，添加需要读取的图片路径）
    parser.add_argument('--image_path', default="2.jpg", help='path of read image')
    # 为解析器添加选项，比如landmarks_model_path，添加特征点定位模型。
    parser.add_argument('--landmarks_model_path', default="shape_predictor_68_face_landmarks.dat", help='path of model')
    # 解析选项的参数
    opt = parser.parse_args()
    # 调用points_face函数，对人脸进行68个点的特征点定位
    points_face(opt)