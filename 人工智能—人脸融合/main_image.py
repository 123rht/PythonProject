# 本文件代码目的：读取图片，进行人脸融合
# 作者：江大白
# 平台网站：www.jiangdabai.com
# 微信公众号1：大白智能
# 微信公众号2：江大白
# 其他：《30天入门人工智能》视频、《国内TOP50大厂&3000篇面经核心秘籍》等等，尽在平台网站查看
# 祝愿：分享的代码和资料，希望对大家入门人工智能有帮助！一路相伴，共同进步！
# 视频教程：本应用涉及很多人脸识别相关的知识点，人脸识别的入门视频，可以查看平台网站的《深入浅出人脸识别基础及项目应用》
#         也涉及到很多图像处理及人脸特效的知识点，人脸特效的入门视频，可以查看平台网站的《深入浅出人脸特效之Mask实战应用》
# 注意：本目录下有一个PDF文件《人脸融合实战详细流程》：描述了详细的人脸融合流程！！！

### 导入封装好的一些模块，直接使用已经写好的函数，简单高效
import cv2  # 图像处理的opencv库
import dlib  # 加载dlib算法模块
import numpy as np  # 导入numpy库，便于数值转换
import argparse  # 加载解析模块

# 加载dlib自带的人脸检测模型
detector = dlib.get_frontal_face_detector()
# 加载dlib的特征点定位模型
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

### 人脸检测及关键点定位函数
def landmark_68(img):
    # 对图像进行人脸检测
    faces = detector(img, 0)
    # 如果len(faces)>0，则证明图片中有人脸
    if len(faces) > 0:
        # 定义一个空矩阵
        landmarks = []
        # 图片中可能有多张人脸，对每张人脸依次进行处理
        for i, rect in enumerate(faces):
            # 对人脸进行关键点定位
            face_lanmarks = landmark_predictor(img, rect)
            # 定义一个68个点的全0矩阵
            landmark = [(0, 0)] * 68
            # 对每个点都提取x、y位置信息
            for j in range(68):
                landmark[j] = (face_lanmarks.part(j).x, face_lanmarks.part(j).y)
            landmarks.append(landmark)
    return landmarks

### 掩膜操作函数
def mask(img, pnt):
    # 新建np的zeros矩阵，比如图像尺寸为（1000,800,3）,新建一个（1000,800）的全零矩阵，便于下面绘制
    mask = np.zeros(img.shape[:2], np.uint8)
    # pnts为landmark中第17个点，第6个点，第26个点，第10个点的集合，近似于人脸的外边界
    # 注意：具体的点的位置信息，可以运行本目录的main_68_points.py，运行后可以看到68个点在人脸部的各个位置
    pnts = np.float32([pnt[17], pnt[6], pnt[26], pnt[10]])
    # chain为需要抠出的人脸区域的各个顶点
    chain = np.array(pnt[0:17] + pnt[24:27][::-1] + pnt[17:20][::-1])
    # 根据各个顶点做凸边型填充，得到一个背景为黑色（0），脸部区域为白色（255）的图像
    mask = cv2.fillConvexPoly(mask, chain, (255, 255, 255))
    # 对图像img和mask0进行按位与操作，黑色背景的区域仍然是黑色，脸部区域不为0，则可以显示出来
    # 最终的效果，即黑色背景，脸部区域为彩色图像的图片
    face = cv2.bitwise_and(img, img, mask=mask)
    return face, pnts, mask

### 对两个图片进行人脸融合
### 注意：因为人脸融合特效的流程中，涉及到很多图像处理操作，会比较复杂
###      因此大白将每一步的操作的效果图，都保存到本目录的image_draw文件夹中，大家可以对应查看图片名，更清楚每一步的效果
###      此外，下面代码中，都有已经注释的cv2.imwrite函数，可以保存每一步的效果图片，大家也可以取消注释，方便查看
def fuse(img, img_add):
    # 获得被人脸融合的图像的高h_img，宽w_img
    h_img, w_img = img.shape[0], img.shape[1]

    # 第一阶段：抠取图像中局部人脸区域
    # 第一步：使用landmark_68函数，对图像中的人脸，进行人脸检测及关键点定位(68个点)
    landmarks_img = landmark_68(img)
    landmarks_img_add = landmark_68(img_add)

    # 图像中可能有多张人脸需要被融合，利用for循环操作，依次遍历每张人脸的信息
    for i in range(len(landmarks_img)):

        # 第二步：使用mask函数，对用来人脸融合的图像，进行掩膜操作
        # 返回的结果中，face为人脸RGB区域的小图，pts为68个点中，[6、10、17、26]四个点构成的源矩阵，近似于人脸的外边界，mask为掩膜人脸区域
        # 注意：本目录的main_68_points.py，运行后可以看到68个点在人脸部的各个位置
        face_img_add, pts_img_add, mask_img_add = mask(img_add, landmarks_img_add[0])
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("face_img_add.jpg",face_img_add)
        # cv2.imwrite("mask_img_add.jpg", mask_img_add)

        # 第三步：使用mask函数，对被人脸融合的每一张人脸，进行掩膜操作
        face_img, pts_img, mask_img = mask(img, landmarks_img[i])
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("face_img.jpg", face_img)
        # cv2.imwrite("mask_img.jpg", mask_img)

        # 第二阶段：局部人脸图像仿射变换
        # 第一步：通过透视变换，计算得到pts_img_add（源矩形）到pts_img(目标矩形)的变换矩形
        M1 = cv2.getPerspectiveTransform(pts_img_add, pts_img)

        # 第二步：将rgb人脸区域，通过变换矩阵，转换到被人脸融合的目标区域
        rgb_warp = cv2.warpPerspective(face_img_add, M1, (w_img, h_img))
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("rgb_warp.jpg", rgb_warp)

        # 第三步：将mask人脸掩膜区域，根据变换矩阵，也转换到被人脸融合的目标区域
        mask_warp = cv2.warpPerspective(mask_img_add, M1, (w_img, h_img))
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("mask_warp.jpg", mask_warp)

        # 第三阶段：人脸区域图像融合
        # 第一步：对mask人脸掩膜区域，使用取反函数，进行取反操作，白色255变为0，黑色0变成255
        mask_inv_warp = cv2.bitwise_not(mask_warp)
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("mask_inv_warp.jpg", mask_inv_warp)

        # 第二步：将mask_inv_warp与img进行按位与处理，即在被人脸融合的图像，抠取相应的区域，等待被人脸融合
        dst1 = cv2.bitwise_and(img, img, mask=mask_inv_warp)
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("dst1.jpg", dst1)

        # 第三步：将等待被人脸融合的图片，和可以融合的rgb小图进行相加，得到完整的贴合图像
        dst2 = cv2.add(rgb_warp, dst1)
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("dst2.jpg", dst2)

        # 第四阶段：人脸边缘无缝融合
        # 第一步：绘制mask_warp掩膜区域的轮廓，通过findContours，获得hierarchy1轮廓的点的集合，rr1轮廓对应的属性
        hierarchy1, rr1 = cv2.findContours(mask_warp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 如需查看，上一行的求取轮廓的效果，可将下面两行前面的#取消，运行后，保存到本目录查看
        # mask_warp_Countours = cv2.drawContours(mask_warp, hierarchy1, -1, (125, 255, 255), 3)
        # cv2.imwrite("mask_warp_Countours.jpg",mask_warp_Countours)

        # 第二步：根据所有轮廓点，计算最小外接矩形，并计算出外接矩形的中心点pointer_center
        for c1 in hierarchy1:
            # 对所有的轮廓c1求一个最小外接矩形，x0,y0,w,h为所对应的矩形的最上方坐标x，y,宽和高
            x0, y0, w, h = cv2.boundingRect(c1)
        point_center = (x0 + w // 2, y0 + h // 2)

        # 第三步：由于直接贴合，分界区域有分界线，看起来比较突兀，因此用cv2.seamlessClone进行修复
        # 注意：无缝融合函数中，dst2是子图，img是母图，mask_warp为掩膜区域，point1为最小外接矩阵，cv2.NORMAL_CLONE为掩膜的类型
        img = cv2.seamlessClone(dst2, img, mask_warp, point_center, cv2.NORMAL_CLONE)
        # 如需查看，上一行的效果，可将cv2.imwrite前面的#取消，运行后，保存到本目录查看
        # cv2.imwrite("img.jpg", img)

    return img

# 人脸融合函数
def fuse_face(opt):
    # 使用opencv中的imread函数，读取被人脸融合的图像
    img = cv2.imread(opt.image_path)
    # 读取用来人脸融合的图像
    img_add = cv2.imread(opt.image_add_path)
    # 使用fuse_image函数，进行人脸融合
    img_result = fuse(img, img_add)
    # 很多时候，图片的长宽太大，屏幕显示不全
    # 为了便于查看图片，可以将图片的大小，缩放到某一尺寸，比如（600,800），即宽600像素，高800像素
    img_result = cv2.resize(img_result, (400, 600))
    # 显示图片
    cv2.imshow("image", img_result)
    # 显示图片停顿的时间，如果是0，则一直显示。如果是100，则显示100ms
    cv2.waitKey(0)

### 函数主入口
if __name__ == '__main__':
    # 新建一个解析器
    parser = argparse.ArgumentParser()
    # 为解析器添加选项，比如image_path，即被人脸融合的图像，可以更换其他图片尝试。（在default后面，添加待人脸融合图片的路径）
    parser.add_argument('--image_path', default="5.png", help='path of read image')
    # 为解析器添加选项，比如image_add_path，即用来人脸融合的图像，可以更换其他图片尝试。（在default后面，添加用来人脸融合图片的路径）
    parser.add_argument('--image_add_path', default="6.jpg", help='path of read image')
    # 为解析器添加选项，比如landmarks_model_path_path，即特征点定位模型的地址。（在default后面，添加需要读取的模型路径）
    parser.add_argument('--landmarks_model_path', default="shape_predictor_68_face_landmarks.dat", help='path of model')
    # 解析选项的参数
    opt = parser.parse_args()
    # 调用fuse_face函数，对图像进行人脸融合
    fuse_face(opt)