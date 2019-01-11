package com.example.qbar;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("iconv");
        System.loadLibrary("zbarjni");
    }

    private TextView mTextView;
    private ImageView mImageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mTextView = findViewById(R.id.text);
        mImageView = findViewById(R.id.image);

        String rerult = scan(R.drawable.test2);
        mTextView.setText(rerult);

//        scan(R.drawable.test1);

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test1);
        BitmapHolder holder = getBitmapSize(bitmap);
        int[] pixels = threshold(holder.pixels, holder.width, holder.height);
        ImageView imageView1 = findViewById(R.id.image1);
        bitmap = Bitmap.createBitmap(pixels, holder.width, holder.height,
                Bitmap.Config.ARGB_8888);
        imageView1.setImageBitmap(bitmap);
    }

    private String scan(int id) {
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), id);
        mImageView.setImageBitmap(bitmap);

        BitmapHolder holder = getBitmapSize(bitmap);
        return decode(holder.pixels, holder.width, holder.height);
    }

    private BitmapHolder getBitmapSize(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        Math.pow(2, 1);
        bitmap.getPixels(pixels, 0, width, 1, 1, width - 1, height - 1);

        BitmapHolder holder = new BitmapHolder();
        holder.width = width;
        holder.height = height;
        holder.pixels = pixels;
        return holder;
    }

    class BitmapHolder {
        public int width;
        public int height;
        public int[] pixels;
    }

    public native String decode(int data[], int width, int height);

    public native int[] gray(int data[], int width, int height);

    public native int[] threshold(int data[], int width, int height);
}
