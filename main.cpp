#include <iostream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>

//#include "histogram.h"
// push
// version 1.0
// version 1.1

// desktop local
// desktop local



// #include "colorDetect.h"
using namespace std;
using namespace cv;

float loc_transform [169][5]={{240,0,0,0,0},
                              {240,0,0,0,1},
                              {240,80,0,0,2},
                              {240,80,0,0,3},
                              {240,160,0,0,4},
                              {240,160,0,0,5},
                              {240,240,0,0,6},
                              {80,0,240,0,7},
                              {80,0,240,0,8},
                              {80,0,160,0,9},
                              {80,0,80,0,10},
                              {80,0,80,0,11},
                              {80,0,0,0,12},
                            {240,0,0,1,0},
                            {240,0,80,1,1},
                            {240,80,80,1,2},
                            {240,80,80,1,3},
                            {240,160,80,1,4},
                            {240,160,80,1,5},
                            {240,240,80,1,6},
                            {80,0,240,1,7},
                            {80,0,240,1,8},
                            {80,0,160,1,9},
                            {80,0,80,1,10},
                            {80,0,80,1,11},
                            {80,0,0,1,12},
                              {240,0,0,2,0},
                              {240,0,0,2,1},
                              {240,80,0,2,2},
                              {240,80,0,2,3},
                              {240,160,0,2,4},
                              {240,160,0,2,5},
                              {240,240,0,2,6},
                              {80,80,240,2,7},
                              {80,80,240,2,8},
                              {80,80,240,2,9},
                              {80,80,160,2,10},
                              {80,80,80,2,11},
                              {80,80,80,2,12},
                               {240,0,160,3,0},
                               {240,0,160.3,1},
                               {240,80,160,3,2},
                               {240,80,160,3,3},
                               {240,160,160,3,4},
                               {240,160,160,3,5},
                               {240,240,160,3,6},
                               {80,80,240,3,7},
                               {80,80,240,3,8},
                               {80,80,160,3,9},
                               {80,80,80,3,10},
                               {80,80,80,3,11},
                               {80,80,0,3.12},
                              {240,0,240,4,0},
                              {240,0,240,4,1},
                              {240,80,240,4,2},
                              {240,80,240,4,3},
                              {240,160,240,4,4},
                              {240,160,240,4,5},
                              {240,240,240,4,6},
                              {80,160,240,4,7},
                              {80,160,240,4,8},
                              {80,160,160,4,9},
                              {80,160,80,4,10},
                              {80,160,80,4,11},
                              {80,160,0,4,12},
                              {240,0,240,5,0},
                              {240,0,240,5,1},
                              {240,80,240,5,2},
                              {240,80,240,5,3},
                              {240,160,240,5,4},
                              {240,160,240,5,5},
                              {240,240,240,5,6},
                              {80,160,240,5,7},
                              {80,160,240,5,8},
                              {80,160,160,5,9},
                              {80,160,80,5,10},
                              {80,160,80,5,11},
                              {80,160,0,5,12},
                                  {0,240,0,6,0},
                                  {0,240,80,6,1},
                                  {0,240,80,6,2},
                                  {0,240,160,6,3},
                                  {0,240,240,6,4},
                                  {0,240,240,6,5},
                                  {0,0,0,6,6},
                                  {80,240,240,6,7},
                                  {80,240,240,6,8},
                                  {80,240,160,6,9},
                                  {80,240,80,6,10},
                                  {80,240,80,6,11},
                                  {80,240,0,6,12},
                               {0,160,0,7,0},
                               {0,160,80,7,1},
                               {0,160,80,7,2},
                               {0,160,160,7,3},
                               {0,160,240,7,4},
                               {0,160,240,7,5},
                               {160,240,240,7,6},
                               {160,160,240,7,7},
                               {160,160,240,7,8},
                               {160,80,240,7,9},
                               {160,80,240,7,10},
                               {160,0,240,7,11},
                               {160,0,240,7,12},
                              {0,160,0,8,0},
                              {0,160,80,8,1},
                              {0,160,80,8,2},
                              {0,160,160,8,3},
                              {0,160,240,8,4},
                              {0,160,240,8,5},
                              {160,240,240,8,6},
                              {160,160,240,8,7},
                              {160,160,240,8,8},
                              {160,80,240,8,9},
                              {160,80,240,8,10},
                              {160,0,240,8,11},
                              {160,0,240,8,12},
                              {0,80,0,9,0},
                              {0,80,80,9,1},
                              {0,80,80,9,2},
                              {0,80,160,9,3},
                              {0,80,240,9,4},
                              {0,80,240,9,5},
                              {160,240,160,9,6},
                              {160,160,160,9,7},
                              {160,160,160,9,8},
                              {160,80,160,9,9},
                              {160,80,160,9,10},
                              {160,0,160,9,11},
                              {160,0,160,9,12},
                              {0,80,0,10,0},
                              {0,80,80,10,1},
                              {0,80,80,10,2},
                              {0,80,160,10,3},
                              {0,80,240,10,4},
                              {0,80,240,10,5},
                              {160,240,80,10,6},
                              {160,160,80,10,7},
                              {160,160,80,10,8},
                              {160,80,80,10,9},
                              {160,80,80,10,10},
                              {160,0,80,10,11},
                              {160,0,80,10,12},
                              {0,0,0,11,0},
                              {0,0,80,11,1},
                              {0,0,80,11,2},
                              {0,0,160,11,3},
                              {0,0,240,11,4},
                              {0,0,240,11,5},
                              {160,240,80,11,6},
                              {160,160,80,11,7},
                              {160,160,80,11,8},
                              {160,80,80,11,9},
                              {160,80,80,11,10},
                              {160,0,80,11,11},
                              {160,0,80,11,12},
                              {0,0,0,12,0},
                              {0,0,80,12,1},
                              {0,0,80,12,2},
                              {0,0,160,12,3},
                              {0,0,240,12,4},
                              {0,0,240,12,5},
                              {160,240,80,12,6},
                              {160,160,80,12,7},
                              {160,160,80,12,8},
                              {160,80,80,12,9},
                              {160,80,80,12,10},
                              {160,0,80,12,11},
                              {160,0,80,12,12}};


void sharpenImage1(const Mat &image, Mat &result){
     //创建并初始化滤波模板
     Mat kernel(3,3,CV_32F,Scalar(0));
     kernel.at<float>(1,1) = 5.0;
     kernel.at<float>(0,1) = -1.0;
     kernel.at<float>(1,0) = -1.0;
     kernel.at<float>(1,2) = -1.0;
     kernel.at<float>(2,1) = -1.0;

     result.create(image.size(),image.type());

     //对图像进行滤波
     filter2D(image,result,image.depth(),kernel);
}



// compute pixel average in center
Scalar avgPixel_BGR(Mat& image){
    int nr = image.rows;
    int nc = image.cols;
    int nr_start = nr*0.1;
    int nc_start = nc*0.1;
    Scalar sum,avg;
    sum.val[0] = 0;
    sum.val[1] = 0;
    sum.val[2] = 0;
//    if(image.isContinuous()){
//        nc = nc*nr;
//        nr =1;
//    }
//    cout << (int)image.at<Vec3b>(10,10)[0] << endl;
    for(int m=nr_start;m<nr-nr_start;m++){
//        uchar* data = image.ptr<uchar>(0);
//        cout << (int)data[10] << endl;
        for(int n=nc_start;n<nc-nc_start;n++){
            sum.val[0] += (int)image.at<Vec3b>(m,n)[0];
            sum.val[1] += (int)image.at<Vec3b>(m,n)[1];
            sum.val[2] += (int)image.at<Vec3b>(m,n)[2];
//            sum.val[0] += (int)data[n*3];
//            sum.val[1] += (int)data[n*3+1];
//            sum.val[2] += (int)data[n*3+2];
        }
    }
    avg.val[0] = sum.val[0]/((nr-nr_start-nr_start)*(nc-nc_start-nc_start));
    avg.val[1] = sum.val[1]/((nr-nr_start-nr_start)*(nc-nc_start-nc_start));
    avg.val[2] = sum.val[2]/((nr-nr_start-nr_start)*(nc-nc_start-nc_start));
//    cout << image.rows << endl;cout << image.rows << endl;
//    cout << avg.val[0] << endl;cout << avg.val[1] << endl;cout << avg.val[2] << endl;
    return avg;
}

Scalar Pixel_BGR(Mat& image){
    int nr = image.rows;
    int nc = image.cols;
    Scalar pixel;

    pixel.val[0] = image.at<Vec3b>(nr*0.5,nc*0.5)[0];
    pixel.val[1] = image.at<Vec3b>(nr*0.5,nc*0.5)[1];
    pixel.val[2] = image.at<Vec3b>(nr*0.5,nc*0.5)[2];
//    cout << image.rows << endl;cout << image.rows << endl;
//    cout << avg.val[0] << endl;cout << avg.val[1] << endl;cout << avg.val[2] << endl;
    return pixel;

}


void ContrastStretch(Mat& image){
    int nr = image.rows;
    int nc = image.cols;
    int data_max=0,data_min=255;
    if(image.isContinuous()){
        nc = nc*nr;
        nr =1;
    }
//    minMaxLoc(image,&data_min,&data_max,0,0);
    for(int m=0;m<nr;m++){
        uchar* data = image.ptr<uchar>(0);
        for(int n=0;n<nc;n++){
            if(data[n]>data_max)
                data_max = data[n];
            if(data[n]<data_min)
                data_min = data[n];
        }
    }
    int data_range = data_max - data_min;
    for(int m=0;m<nr;m++){
        uchar* data = image.ptr<uchar>(0);
        for(int n=0;n<nc;n++){
            data[n] = (data[n] - data_min)*255/data_range;
        }
    }
}

void ContrastStretchRGB(Mat& image){
    int nr = image.rows;
    int nc = image.cols*image.channels();
    int data_maxR=0,data_minR=255;
    int data_maxG=0,data_minG=255;
    int data_maxB=0,data_minB=255;
    if(image.isContinuous()){
        nc = nc*nr;
        nr =1;
    }
    for(int m=0;m<nr;m++){
        uchar* data = image.ptr<uchar>(0);
        for(int n=0;n<nc;n++){
            if(data[n]>data_maxB)
                data_maxB = data[n];
            if(data[n]<data_minB)
                data_minB = data[n];
            n=n+1;
            if(data[n]>data_maxG)
                data_maxG = data[n];
            if(data[n]<data_minG)
                data_minG = data[n];
            n=n+1;
            if(data[n]>data_maxR)
                data_maxR = data[n];
            if(data[n]<data_minR)
                data_minR = data[n];
        }
    }
    int data_rangeR = data_maxR - data_minR;
    int data_rangeG = data_maxG - data_minG;
    int data_rangeB = data_maxR - data_minB;

    for(int m=0;m<nr;m++){
        uchar* data = image.ptr<uchar>(0);
        for(int n=0;n<nc;n++){
            data[n] = (data[n] - data_minB)*255/data_rangeB;
            n=n+1;
            data[n] = (data[n] - data_minG)*255/data_rangeG;
            n=n+1;
            data[n] = (data[n] - data_minR)*255/data_rangeR;
        }
    }
}

// compute adaptive ROI
void adaptiveimgROI(Mat& image, Mat&result, double hight){



}


void computeROI(Mat& image,Mat& imageROI, double h){
    int rect_length,rect_width,length_begin,width_begin;
    rect_length = (int)120/h; //parameter estimate
    rect_width = (int)120/h; //4:3
    length_begin = (int)(640-rect_length)/2;
    width_begin = (int)(426-rect_width)/2;

    imageROI = image(Rect(length_begin,width_begin,rect_length,rect_width));

}


int main(int argc, char *argv[])
{
        Mat img = imread("/home/dysen/work_opencv/opencv_capture/6.jpg",CV_LOAD_IMAGE_COLOR);
//        cout << img.rows << img.cols << endl;
        Mat imgROI;
        computeROI(img,imgROI,1);
//        cout << imgROI.rows << imgROI.cols << endl;
        imshow("loc4",imgROI);
//        ContrastStretchRGB(img);
//        imshow("loc4con",img);

//        Mat imgROI = img(Rect(270,270,100,100));
//        imshow("aa",imgROI);

//        int minHessian =400;
//        SurfFeatureDetector detector(minHessian);
//        SurfFeatureDetector (SURF);
//        vector<KeyPoint> keypoints_1;
//        detector.detect(img,keypoints_1);

//        Mat img_keypoints_1;
//        drawKeypoints(img,keypoints_1,img_keypoints_1, Scalar::all(-1),DrawMatchesFlags::DEFAULT);

//        imshow("img_keypoints_1",img_keypoints_1);
        Mat dst;
        dst.create(imgROI.size(),imgROI.type());
        Mat imggray;
        cvtColor(imgROI,imggray,CV_BGR2GRAY);
//        imshow("imggray11111",imggray);

        ContrastStretch(imggray);

//        threshold(imggray,imggray,160,250,THRESH_BINARY);
//        adaptiveThreshold(imggray,imggray,250,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,5,5);
        imshow("imggray",imggray);
        Mat res;

        blur(imggray,res,Size(3,3));      
//        GaussianBlur(imggray,imggray,Size(7,7),1.5,1.5);

        Canny(res,res,20,50);

//        Canny(imggray,res,30,75);
//        Canny(imggray,res,40,100);
//        Canny(imggray,res,50,125);
//        Canny(imggray,res,60,150);
//        Canny(imggray,res,70,175);
//        Canny(imggray,res,80,200);

        dst = Scalar::all(0);
        imgROI.copyTo(dst,res);
        imshow("bbb",dst);

        vector<vector<Point> > contours;
        findContours(res,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

        // compute midian block area
        double area_median;
        double avg_locx=0,avg_locy=0;
        int num=0;

        if(contours.size()%2 ==0){
            Rect rect_median_1 = boundingRect(contours[(int)(contours.size()/2-1)]);
            Rect rect_median_2 = boundingRect(contours[(int)(contours.size()/2)]);
            area_median = (rect_median_1.area()+rect_median_2.area()) / 2;
        }
        else{
            Rect rect_median = boundingRect(contours[(int)(contours.size()/2)]);
            area_median = rect_median.area();
        }
        cout << area_median << endl;
        int area=0;
        for(size_t k=0;k<contours.size();k++){
             Rect rect_tmp =  boundingRect(contours[k]);
//             RotatedRect  rorect = minAreaRect(contours[k]);
             Mat image_tmp;
             image_tmp = img(rect_tmp);
             Scalar average;
             average = avgPixel_BGR(image_tmp);
             cout << "pixelB is : " << average.val[0]
                  << " pixelG is : " << average.val[1]
                  << " pixelR is : " << average.val[2] << endl;

//             if(abs(rect_tmp.area()-area_median)<100){
                 for(int j =0;j<169;j++){
                     if((abs(average.val[2] - loc_transform[j][0])<30) &&
                        (abs(average.val[1] - loc_transform[j][1])<30) &&
                        (abs(average.val[0] - loc_transform[j][2])<30)){
                         cout << "locationx is:" << loc_transform[j][3]
                                 <<" locationy is:" << loc_transform[j][4] << endl;
                         avg_locx += loc_transform[j][3];
                         avg_locy += loc_transform[j][4];
                         num++;
//                         break;
                     } 
                 }

//                 imshow("img1",image1);
//                 if(rect.area()>100 && rect.area()<300)
                 cout << "area: " << ++area << ' ' << rect_tmp.area() << endl;
//            }
        }
        if(num!=0){
           avg_locx = avg_locx/num;
           avg_locy = avg_locy/num;
           cout << "avg_locx is:" << avg_locx
                <<" avg_locy is:" << avg_locy << endl;
        }
//        imshow("res", contours);

        Mat  result(res.size(),CV_8U,Scalar(255));
        drawContours(result,contours,-1,Scalar(0),2);
//        imshow("resultImage",result);
        waitKey();
        return 0;


//        cvtColor(img, img, CV_BGR2GRAY);
////        imshow("aaa",img);
//        Mat res;
//        clock_t system_start = clock();

//        Canny(img, res, 50, 110);
////        imshow("aaa",res);

//        clock_t diff = clock() - system_start;
//        cout << "output time" << diff << endl;
}
