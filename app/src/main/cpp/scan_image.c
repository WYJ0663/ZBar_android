#include <stdio.h>
#include <stdlib.h>
#include <zbar.h>

//#if !defined(PNG_LIBPNG_VER) || \
//    PNG_LIBPNG_VER < 10018 || \
//    (PNG_LIBPNG_VER > 10200 && \
//     PNG_LIBPNG_VER < 10209)
/* Changes to Libpng from version 1.2.42 to 1.4.0 (January 4, 2010)
 * ...
 * 2. m. The function png_set_gray_1_2_4_to_8() was removed. It has been
 *       deprecated since libpng-1.0.18 and 1.2.9, when it was replaced with
 *       png_set_expand_gray_1_2_4_to_8() because the former function also
 *       expanded palette images.
 */
# define png_set_expand_gray_1_2_4_to_8 png_set_gray_1_2_4_to_8
#endif

zbar_image_scanner_t *scanner = NULL;

/* to complete a runnable example, this abbreviated implementation of
 * get_data() will use libpng to read an image file. refer to libpng
 * documentation for details
 */

int scan_image(int width, int height, void *raw) {
    /* create a reader */
    scanner = zbar_image_scanner_create();

    /* configure the reader */
    zbar_image_scanner_set_config(scanner, 0, ZBAR_CFG_ENABLE, 1);

    /* obtain image data */
//    int width = 0, height = 0;
//    void *raw = NULL;

    /* wrap image data */
    zbar_image_t *image = zbar_image_create();
    zbar_image_set_format(image, zbar_fourcc('Y', '8', '0', '0'));
    zbar_image_set_size(image, width, height);
    zbar_image_set_data(image, raw, width * height, zbar_image_free_data);

    /* scan the image for barcodes */
    int n = zbar_scan_image(scanner, image);

    /* extract results */
    const zbar_symbol_t *symbol = zbar_image_first_symbol(image);
    for (; symbol; symbol = zbar_symbol_next(symbol)) {
        /* do something useful with results */
        zbar_symbol_type_t typ = zbar_symbol_get_type(symbol);
        const char *data = zbar_symbol_get_data(symbol);
        printf("decoded %s symbol \"%s\"\n",
               zbar_get_symbol_name(typ), data);
    }

    /* clean up */
    zbar_image_destroy(image);
    zbar_image_scanner_destroy(scanner);

    return (0);
}
