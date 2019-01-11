//
// Created by yijunwu on 2019/1/11.
//

#include <malloc.h>
#include "color_util.h"


int alpha(int color) {
    return (color >> 24) & 0xFF;
}

int red(int color) {
    return (color >> 16) & 0xFF;
}

int green(int color) {
    return (color >> 8) & 0xFF;
}

int blue(int color) {
    return color & 0xFF;
}

char *get_gray_color(jint *data, int width, int height) {
    char *gray = malloc(width * height);
    int i = 0;
    int j = 0;
    for (i = 0; i < width; i++) {
        for (j = 0; j < height; j++) {
            int curr_color = data[j * width + i];
            int r = red(curr_color);
            int g = green(curr_color);
            int b = blue(curr_color);
            char color = (char) (r * 0.3 + g * 0.59 + b * 0.11);
            gray[j * width + i] = color;
        }
    }
    return gray;
}

char *binarization(jint *data, int width, int height) {
    char *out = malloc(width * height);
    int i = 0;
    int j = 0;
    for (i = 0; i < width; i++) {
        for (j = 0; j < height; j++) {
            int curr_color = data[j * width + i];
            int r = red(curr_color);
            int g = green(curr_color);
            int b = blue(curr_color);
            if((r+g+b)/3 >127){
                out[j * width + i] = (char) 0xff;
            } else{
                out[j * width + i] = 0x00;
            }

        }
    }
    return out;
}

