#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "log.h"

using namespace cv;
using namespace std;


//测试使用，编译要注释掉
//#include <stl/_vector.h>

/** Function Headers */
Mat detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "/sdcard/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/sdcard/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);

IplImage *change4channelTo3InIplImage(IplImage *src);


/* angle: finds a cosine of angle between vectors, from pt0->pt1 and from pt0->pt2
 */
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
           sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

/* findLargestSquare: find the largest square within a set of squares
 */
void findLargestSquare(const vector<vector<Point>> &squares,
                       vector<Point> &biggest_square) {
    if (!squares.size()) {
        LOGD("%s", "findLargestSquare !!! No squares detect, nothing to do.");
        return;
    }

    int max_width = 0;
    int max_height = 0;
    int max_square_idx = 0;
    for (size_t i = 0; i < squares.size(); i++) {
        // Convert a set of 4 unordered Points into a meaningful cv::Rect structure.
        Rect rectangle = boundingRect(Mat(squares[i]));

        //std::cout << "find_largest_square: #" << i << " rectangle x:" << rectangle.x << " y:" << rectangle.y << " " << rectangle.width << "x" << rectangle.height << endl;

        // Store the index position of the biggest square found
        if ((rectangle.width >= max_width) && (rectangle.height >= max_height)) {
            max_width = rectangle.width;
            max_height = rectangle.height;
            max_square_idx = i;
        }
    }

    biggest_square = squares[max_square_idx];
}

extern "C" {

JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_gray(
        JNIEnv *env, jclass obj, jintArray buf, int w, int h) {

    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat imgData(h, w, CV_8UC4, (unsigned char *) cbuf);

    uchar *ptr = imgData.ptr(0);
    for (int i = 0; i < w * h; i++) {
        //计算公式：Y(亮度) = 0.299*R + 0.587*G + 0.114*B
        //对于一个int四字节，其彩色值存储方式为：BGRA
        int grayScale = (int) (ptr[4 * i + 2] * 0.299 + ptr[4 * i + 1] * 0.587 +
                               ptr[4 * i + 0] * 0.114);
        ptr[4 * i + 1] = grayScale;
        ptr[4 * i + 2] = grayScale;
        ptr[4 * i + 0] = grayScale;
    }

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, cbuf);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}
////////////////////////////////

JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_canny(
        JNIEnv *env, jclass type, jintArray buf, jint w, jint h) {

    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat myimg(h, w, CV_8UC4, (unsigned char *) cbuf);//转换图片数据

//    Mat cimg;
//    cvCanny(myimg, cimg, 50, 150, 3);
//
//    jint *ptr = (jint *) cimg.ptr(0);
//
//    int size = w * h;
//    jintArray result = env->NewIntArray(size);
//    env->SetIntArrayRegion(result, 0, size, ptr);
//    env->ReleaseIntArrayElements(buf, cbuf, 0);
//    return result;

    IplImage image = IplImage(myimg);
    IplImage *image3channel = change4channelTo3InIplImage(&image);

    IplImage *pCannyImage = cvCreateImage(cvGetSize(image3channel), IPL_DEPTH_8U, 1);

    cvCanny(image3channel, pCannyImage, 50, 150, 3);

    int *outImage = new int[w * h];
    for (int i = 0; i < w * h; i++) {
        outImage[i] = (int) pCannyImage->imageData[i];
    }

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, outImage);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

/////////2222222222222222222222222222222///////////////
JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_gray2(
        JNIEnv *env, jclass type, jintArray buf, jint w, jint h) {


    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);

    Mat grayImage;
    cvtColor(srcImage, grayImage, COLOR_BGRA2GRAY);
    cvtColor(grayImage, grayImage, COLOR_GRAY2BGRA);
    jint *ptr = grayImage.ptr<jint>(0);

    int size = w * h;

    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_erode(JNIEnv *env, jclass type, jintArray buf, jint w, jint h) {
    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);

    //腐蚀算法
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat dstImage;
    erode(srcImage, dstImage, element);

    jint *ptr = (jint *) dstImage.ptr(0);


    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_open(JNIEnv *env, jclass type, jintArray buf, jint w, jint h) {
    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);


    //开运算
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    //进行形态学操作
    Mat dstImage;
    morphologyEx(srcImage, dstImage, MORPH_OPEN, element);


    jint *ptr = (jint *) dstImage.ptr(0);


    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

//闭运算
JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_close(JNIEnv *env, jclass type, jintArray buf, jint w, jint h) {
    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);

    //运算
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    //进行形态学操作
    Mat dstImage;
    morphologyEx(srcImage, dstImage, MORPH_CLOSE, element);

    jint *ptr = (jint *) dstImage.ptr(0);

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;

}

//二值图
JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_binary(
        JNIEnv *env, jclass type, jintArray buf, jint w, jint h) {
    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);

    //对灰度图进行阈值操作，得到二值图并显示
    Mat harrisCorner;
    threshold(srcImage, harrisCorner, 0.00001, 255, THRESH_BINARY);

    jint *ptr = (jint *) harrisCorner.ptr(0);

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

//高斯模糊
JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_gaussianBlur
        (JNIEnv *env, jclass type, jintArray buf, jint w, jint h, jint s) {
    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);
    Mat outImage;

    GaussianBlur(srcImage, outImage, Size(s, s), 0, 0);

    jint *ptr = (jint *) outImage.ptr(0);

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_findContours(
        JNIEnv *env, jclass type, jintArray raw, jintArray buf, jint w, jint h) {
    LOGD("%s", "开始");

    jint *craw;
    craw = env->GetIntArrayElements(raw, JNI_FALSE);
    if (craw == NULL) {
        return 0;
    }
    Mat rawImage(h, w, CV_8UC4, (unsigned char *) craw);

    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }
    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);

    Mat grayImage;
    cvtColor(srcImage, grayImage, COLOR_BGRA2GRAY);
//    cvtColor(grayImage, grayImage, COLOR_GRAY2BGRA);

//    Mat outImage;

    vector<vector<Point>> squares;

    // Find contours and store them in a list
    vector<vector<Point>> contours;
    findContours(grayImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    LOGD("contours size %d ", contours.size());
//    LOGD("contours %d", contours.size());

    // Test contours and assemble squares out of them
    vector<Point> approx;
    for (size_t i = 0; i < contours.size(); i++) {
        // approximate contour with accuracy proportional to the contour perimeter
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation
        if (approx.size() == 4 && std::fabs(contourArea(Mat(approx))) > 1000 &&
            isContourConvex(Mat(approx))) {
            double maxCosine = 0;
            for (int j = 2; j < 5; j++) {
                double cosine = std::fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                maxCosine = MAX(maxCosine, cosine);
            }

            if (maxCosine < 0.3)
                squares.push_back(approx);
        }
    }

    LOGD("squares size %d ", squares.size());

    vector<Point> largest_square;
    findLargestSquare(squares, largest_square);

//    vector <Point2f> img_pts(4);

    for (size_t i = 0; i < largest_square.size(); i++) {
        circle(rawImage, largest_square[i], 4, Scalar(0, 0, 255), CV_FILLED);
//        Point point = largest_square[i];
//        img_pts[i] = Point2f(point.x, point.y);
    }


    //  topLeft, topRight, bottomRight, bottomLeft
    //计算点的排序
    vector<Point2f> img_pts(4);
    int max1 = 0;
    int min1 = 0;

    int max = 0;
    int min = 0;

    for (size_t i = 0; i < largest_square.size(); i++) {
        Point point = largest_square[i];
        int t1 = point.x * point.x + point.y * point.y;
        int temX = srcImage.cols - point.x;
        int t2 = temX * temX + point.y * point.y;

        if (t1 + t2 > max1 || max1 == 0) {
            img_pts[3] = Point2f(point.x, point.y);
            max1 = t1 + t2;
            max = i;
        }
        if (t1 + t2 < min1 || min1 == 0) {
            img_pts[0] = Point2f(point.x, point.y);
            min1 = t1 + t2;
            min = i;
        }
    }

    int max2 = 0;
    int min2 = 0;
    for (size_t i = 0; i < largest_square.size(); i++) {
        if (i == max || i == min) {
            continue;
        }

        Point point = largest_square[i];
        int t1 = point.x * point.x + point.y * point.y;
        int temX = srcImage.cols - point.x;
        int t2 = temX * temX + point.y * point.y;

        if (t1 + t2 > max2 || max2 == 0) {
            img_pts[2] = Point2f(point.x, point.y);
            max2 = t1 + t2;
        }

        if (t1 + t2 < min2 || min2 == 0) {
            img_pts[1] = Point2f(point.x, point.y);
            min2 = t1 + t2;
        }
    }

//    LOGD("img_pts size %d max $d min %d", img_pts.size(), max, min);

    for (size_t i = 0; i < img_pts.size(); i++) {
        Point point = img_pts[i];
        LOGD("img_pts %d  x %d y %d", i, point.x, point.y);

//        Point point2 = largest_square[i];
//        LOGD("largest_square %d  x %d y %d", i, point2.x, point2.y);
    }


    int w_a4 = 1280, h_a4 = 720;
    //int w_a4 = 595, h_a4 = 842;
    Mat dstImage = Mat::zeros(h_a4, w_a4, CV_8UC4);
    //  topLeft, topRight, bottomRight, bottomLeft
    // corners of destination image with the sequence [tl, tr, bl, br]
    vector<Point2f> dst_pts(4);
    dst_pts[0] = Point2f(0, 0);
    dst_pts[1] = Point2f(w_a4 - 1, 0);
    dst_pts[2] = Point2f(0, h_a4 - 1);
    dst_pts[3] = Point2f(w_a4 - 1, h_a4 - 1);

    // get transformation matrix
    Mat transmtx = getPerspectiveTransform(img_pts, dst_pts);

    // apply perspective transformation
    warpPerspective(rawImage, dstImage, transmtx, dstImage.size());


//    cvtColor(dst, dst, COLOR_BGR2BGRA);

    jint *ptr = (jint *) dstImage.ptr(0);

    int size = w_a4 * h_a4;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}


JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_transform(JNIEnv *env, jclass type, jintArray buf, jint w,
                                             jint h) {
    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);
    int img_height = srcImage.rows;
    int img_width = srcImage.cols;

    //    int w_a4 = 720, h_a4 = 1280;
//    //int w_a4 = 595, h_a4 = 842;
//    Mat dstImage = Mat::zeros(h_a4, w_a4, CV_8UC4);
    //  topLeft, topRight, , bottomLeft,bottomRight
    vector<Point2f> corners(4);
    corners[0] = Point2f(0, 0);
    corners[1] = Point2f(img_width - 1, 0);
    corners[2] = Point2f(0, img_height - 1);
    corners[3] = Point2f(img_width - 1, img_height - 1);

    vector<Point2f> corners_trans(4);
    corners_trans[1] = Point2f(150, 250);
    corners_trans[0] = Point2f(450, 0);
    corners_trans[2] = Point2f(0, img_height - 1);
    corners_trans[3] = Point2f(400, img_height - 1);

    Mat img_trans = Mat::zeros(img_height, img_width, CV_8UC4);

    Mat transform = getPerspectiveTransform(corners, corners_trans);

    warpPerspective(srcImage, img_trans, transform, img_trans.size());


    jint *ptr = (jint *) img_trans.ptr(0);

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

JNIEXPORT jintArray JNICALL
Java_com_ptwyj_opencv_OpenCVHelper_face(
        JNIEnv *env, jclass type, jintArray buf, jint w, jint h) {
    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }
    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);


//-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name)) {
        LOGD("--(!)Error loading\n");
        return 0;
    };
    if (!eyes_cascade.load(eyes_cascade_name)) {
        LOGD("--(!)Error loading\n");
        return 0;
    };

    Mat outImage = detectAndDisplay(srcImage);

    jint *ptr = (jint *) outImage.ptr(0);
    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}


}


IplImage *change4channelTo3InIplImage(IplImage *src) {
    if (src->nChannels != 4) {
        return NULL;
    }

    IplImage *destImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
    for (int row = 0; row < src->height; row++) {
        for (int col = 0; col < src->width; col++) {
            CvScalar s = cvGet2D(src, row, col);
            cvSet2D(destImg, row, col, s);
        }
    }

    return destImg;
}


/** @function detectAndDisplay */
Mat detectAndDisplay(Mat frame) {
    vector<Rect> faces;
    Mat frame_gray;
    cvtColor(frame, frame_gray, CV_BGRA2GRAY);
    equalizeHist(frame_gray, frame_gray);
//-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t i = 0; i < faces.size(); i++) {
        Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360,
                Scalar(255, 0, 255), 4, 8, 0);
        Mat faceROI = frame_gray(faces[i]);
        vector<Rect> eyes;
//-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (size_t j = 0; j < eyes.size(); j++) {
            Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
                         faces[i].y + eyes[j].y + eyes[j].height * 0.5);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
        }
    }
//-- Show what you got
    return frame;
}
