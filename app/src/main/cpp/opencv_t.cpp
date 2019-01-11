//
// Created by yijunwu on 2019/1/10.
//

#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_qbar_MainActivity_gray(
        JNIEnv *env, jobject instance, jintArray buf, jint w, jint h) {


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


