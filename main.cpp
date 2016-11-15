#include <iostream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>

// push
// version 1.0
// version 1.1

// desktop local
// desktop local 2



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
    Scalar sum,avg;
    sum.val[0] = 0;
    sum.val[1] = 0;
    sum.val[2] = 0;
//    if(image.isContinuous()){
//        nc = nc*nr;
//        nr =1;
//    }
//    cout << (int)image.at<Vec3b>(10,10)[0] << endl;
    for(int m=10;m<nr-10;m++){
//        uchar* data = image.ptr<uchar>(0);
//        cout << (int)data[10] << endl;
        for(int n=10;n<nc-10;n++){
            sum.val[0] += (int)image.at<Vec3b>(m,n)[0];
            sum.val[1] += (int)image.at<Vec3b>(m,n)[1];
            sum.val[2] += (int)image.at<Vec3b>(m,n)[2];
//            sum.val[0] += (int)data[n*3];
//            sum.val[1] += (int)data[n*3+1];
//            sum.val[2] += (int)data[n*3+2];
        }
    }
    avg.val[0] = sum.val[0]/((nr-20)*(nc-20));
    avg.val[1] = sum.val[1]/((nr-20)*(nc-20));
    avg.val[2] = sum.val[2]/((nr-20)*(nc-20));
//    cout << image.rows << endl;
//    cout << image.rows << endl;
//    cout << avg.val[0] << endl;
//    cout << avg.val[1] << endl;
//    cout << avg.val[2] << endl;
    return avg;
}


int main(int argc, char *argv[])
{
        Mat img = imread("/home/dysen/work_opencv/opencv_capture/loc4.jpg",CV_LOAD_IMAGE_COLOR);
//        Mat img = imgread(Rect(270,190,200,200));

//        Mat dstImage, tmpImagel;
//        resize(img,dstImage,Size(img.cols*2, img.rows*2));
//        sharpenImage1(img,dstImage);
//        imshow("loc4pyrDown",dstImage);
        imshow("loc4",img);

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
        dst.create(img.size(),img.type());
        Mat imggray;
        cvtColor(img,imggray,CV_BGR2GRAY);

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
        img.copyTo(dst,res);
        imshow("bbb",dst);

        vector<vector<Point> > contours;
        findContours(res,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

//        Rect rect1 =  boundingRect(contours[1]);
////             RotatedRect  rorect = minAreaRect(contours[k]);
//        Mat image1;
//        image1 = img(rect1);
//        imshow("img1",image1);
//        Rect rect2 =  boundingRect(contours[2]);
////             RotatedRect  rorect = minAreaRect(contours[k]);
//        Mat image2;
//        image2 = img(rect2);
//        imshow("img2",image2);
        for(size_t k=0;k<contours.size();k++){
             Rect rect1 =  boundingRect(contours[k]);
//             RotatedRect  rorect = minAreaRect(contours[k]);
             Mat image1;
             image1 = img(rect1);
             Scalar average;
             average = avgPixel_BGR(image1);
             cout << "pixelB is : " << average.val[0]
                  << " pixelG is : " << average.val[1]
                  << " pixelR is : " << average.val[2] << endl;

//             int pixelB = image1.at<Vec3b>(10,10)[0];
//             int pixelG = image1.at<Vec3b>(10,10)[1];
//             int pixelR = image1.at<Vec3b>(10,10)[2];
//             cout << "pixelB is : " << pixelB
//                  << " pixelG is : " << pixelG
//                  << " pixelR is : " << pixelR << endl;

             for(int j =0;j<169;j++){
                 if((abs(average.val[2] - loc_transform[j][0])<5) &&
                         (abs(average.val[1] - loc_transform[j][1])<5) &&
                         (abs(average.val[0] - loc_transform[j][2])<5)){
                     cout << "locationx is:" << loc_transform[j][3]
                             <<" locationy is:" << loc_transform[j][4] << endl;
//                     break;

                 }

             }

//             imshow("img1",image1);


//             if(rect.area()>100 && rect.area()<300)
             cout << "area: " << k+1 << ' '<< rect1.area() << endl;

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
//        threshold(res, res, 128, 255, THRESH_BINARY);
//        imshow("mysobel", res);
//        waitKey();
//        return 0;
}




//#include "opencv2/core/core.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

//#include <iostream>
//#include <stdio.h>

//using namespace std;
//using namespace cv;

///** Function Headers */
//void detectAndDisplay( Mat frame );

///** Global variables */
////-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
//String face_cascade_name = "haarcascade_frontalface_alt.xml";
//String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//string window_name = "Capture - Face detection";
//RNG rng(12345);

///**
// * @function main
//*/
//int main(int argc, char *argv[])
//{
//  VideoCapture capture(0);
//  Mat frame;


//  //-- 1. Load the cascades
////  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
////  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

//  //-- 2. Read the video stream
////  capture.open( -1 );
//  if( capture.isOpened() )
//  {
//    for(;;)
//    {

//        namedWindow("cam");
//        capture >> frame;
//        imshow("cam",frame);
//        waitKey(30);

////      //-- 3. Apply the classifier to the frame
////      if( !frame.empty() )
////       { detectAndDisplay( frame ); }
////      else
////       { printf(" --(!) No captured frame -- Break!"); break; }

////      int c = waitKey(10);
////      if( (char)c == 'c' ) { break; }

//    }
//  }
//  return 0;
//}

///**
// * @function detectAndDisplay
// */
//void detectAndDisplay( Mat frame )
//{
//   std::vector<Rect> faces;
//   Mat frame_gray;

//   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
//   equalizeHist( frame_gray, frame_gray );
//   //-- Detect faces
//   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

//   for( size_t i = 0; i < faces.size(); i++ )
//    {
//      Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
//      ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );

//      Mat faceROI = frame_gray( faces[i] );
//      std::vector<Rect> eyes;

//      //-- In each face, detect eyes
//      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

//      for( size_t j = 0; j < eyes.size(); j++ )
//       {
//         Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
//         int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//         circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
//       }
//    }
//   //-- Show what you got
//   imshow( window_name, frame );
//}
