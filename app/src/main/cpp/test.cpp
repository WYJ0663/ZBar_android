#include <zbar.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <jni.h>
#include "log.h"

extern "C" {
#include "color_util.h"
}

using namespace cv;
using namespace zbar;

int maincpp(void *raw, unsigned width, unsigned height) {
    LOGE("maincpp");
    // create a reader
    ImageScanner scanner;

    // configure the reader
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

    // wrap image data
    Image image(width, height, "Y800", raw, width * height);

    // scan the image for barcodes
    int n = scanner.scan(image);

    // extract results
    for (Image::SymbolIterator symbol = image.symbol_begin();
         symbol != image.symbol_end();
         ++symbol) {
        // do something useful with results
//        cout << "decoded " << symbol->get_type_name()
//             << " symbol \"" << symbol->get_data() << '"' << endl;
        LOGE("type %s,data %s", symbol->get_type_name().c_str(), symbol->get_data().c_str());
        if (!symbol->get_data().empty()){
            return 1;
        }
    }

    // clean up
    image.set_data(NULL, 0);

    return (0);
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_qbar_MainActivity_threshold(JNIEnv *env, jobject instance, jintArray buf, jint w, jint h) {

    jint *cbuf;
    cbuf = env->GetIntArrayElements(buf, JNI_FALSE);
    if (cbuf == NULL) {
        return 0;
    }

//    char *gray_data = binarization(cbuf, w, h);

    Mat srcImage(h, w, CV_8UC4, (unsigned char *) cbuf);
    //灰度
    Mat grayImage;
    cvtColor(srcImage, grayImage, COLOR_BGRA2GRAY);
//    cvtColor(grayImage, grayImage, COLOR_GRAY2BGRA);

//    对灰度图进行阈值操作，得到二值图并显示
//    Mat harrisCorner;
//    threshold(grayImage, harrisCorner, 127, 255, THRESH_BINARY);

    Mat harrisCorner2;
    threshold(grayImage, harrisCorner2, 127, 255, THRESH_BINARY);
//    double thre = threshold(grayImage, harrisCorner2, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
//    adaptiveThreshold(grayImage, harrisCorner2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21,0);
//    LOGE("thre %d", thre);

//    threshold第一个参数即原图像必须为灰度图,最佳33
//
//    //闭运算
//    //进行形态学操作
//    Mat closeImage;
//    morphologyEx(harrisCorner, closeImage, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(15, 15)));
//
//    //开运算
//    //进行形态学操作
//    Mat openImage;
//    morphologyEx(harrisCorner, openImage, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(15, 15)));

    Mat out;
    cvtColor(harrisCorner2, out, COLOR_GRAY2BGRA);
    jint *ptr = (jint *) out.ptr(0);

//    char *gray_data2 = binarization(ptr, w, h);
    maincpp(harrisCorner2.ptr(0), w, h);

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, ptr);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;
}

//
//int GetQR(Mat img) {
//    Mat binImg;
//    Mat adaptiveImg;
//    //在otsu二值结果的基础上，不断增加阈值，用于识别模糊图像
//    double thre = threshold(img, binImg, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);//threshold 第一个参数即原图像必须为灰度图
//
////    cout << "thre的值" << thre << endl;
//    QRres result;
//    result = GetQRInBinImg(binImg);
//    if (result.res.empty())//如果阈值otsuthreshold失败，则采用高斯自适应阈值化，可以识别出一定的控制阈值也识别不出来的二维码
//    {
//        cout << "高度是" << img.cols << endl;
//        cout << "宽度是" << img.rows << endl;
//        cv::adaptiveThreshold(img, adaptiveImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 33,
//                              0);//threshold第一个参数即原图像必须为灰度图,最佳33
//        //imshow("adaptive", adaptiveImg);
//        waitKey();
//        result = GetQRInBinImg(adaptiveImg);
//        if (!result.res.empty()) {
//            cout << "adaptive.res" << result.res << endl;
//            a[50 + S_count] = "adaptive多成功的2";
//            S_count++;
//        }
//    }
//    cout << "GetQR res的值" << result.res << endl;
//    thre = thre / 2;//ostu和自适应阈值都失败，将从ostu阈值的一般开始控制阈值不断增长
//    while (result.res.empty() && thre < 255) {
//        threshold(img, binImg, thre, 255, cv::THRESH_BINARY);
//        imshow("binImg", binImg);
//        //waitKey();
//        result = GetQRInBinImg(binImg);
//        thre += 5;//阈值步长设为5，步长越大，识别率越低，速度越快,对于当前测试图片的情况为5识别出来的最多
//    }
//    return result;
//}
