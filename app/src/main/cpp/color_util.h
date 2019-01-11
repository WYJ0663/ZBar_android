//
// Created by yijunwu on 2019/1/11.
//

#ifndef QBAR_COLOR_UTIL_H
#define QBAR_COLOR_UTIL_H


#include <jni.h>

char *get_gray_color(jint *data, int width, int height);

char *binarization(jint *data, int width, int height);

#endif //QBAR_COLOR_UTIL_H
