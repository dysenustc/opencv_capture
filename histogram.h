#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <istream>
#include <vector>

using namespace std;
using namespace cv;

class Histogram1D
{
private:
    int histSize[1]; //
    float hranges[2];
    const float* ranges[1];
    int channels[1];

public:
    Histogram1D();
    MatND getHistogram(const Mat& image);
    Mat getHistogramImage(const Mat& image);
    Mat applyLookUp(Mat& image,const Mat& lookup);
    Mat stretch(const Mat& image,int minValue=0);

};


#endif // HISTOGRAM_H

Histogram1D::Histogram1D()
{
    histSize[0] = 256;
    hranges[0] = 0.0;
    hranges[1] = 255.0;
    ranges[0] = hranges;
    channels[0] = 0;
}

MatND Histogram1D::getHistogram(const Mat &image)
{
    MatND hist;
    calcHist(image,channels,Mat(),hist,1,histSize,ranges);
    return hist;
}

Mat Histogram1D::getHistogramImage(const Mat &image)
{
    MatND hist = getHistogram(image);
    double maxVal = 0;
    double minVal = 0;
    minMaxLoc(hist,&minVal,&maxVal,0,0);
    Mat histImg(histSize[0],histSize[0],CV_8U,Scalar(255));
    int hpt = static_cast<int>(0.9*histSize[0]);
    for(int h=0;h<histSize[0];h++){
        float binVal = hist.at<float>(h);
        int intensity = static_cast<int>(binVal*hpt/maxval);
        line(histImg,Point(h,histSize[0]),Point(h,histSize[0]-intensity),Scalar::all(0));

    }
    return histImg;
}

Mat Histogram1D::applyLookUp(Mat &image, const Mat &lookup)
{
    Mat result;
    LUT(image,lookup,result);
    return result;
}

Mat Histogram1D::stretch(const Mat &image, int minValue)
{
    MatND hist = getHistogram(image);
    int imin = 0;
    for(;imin<histSize[0];imin++){
        cout << hist.at<float>(imin) << endl;
        if(hist.at<float>(imin) > minValue)
            break;
    }
    int imax = histSize[0]-1;
    for(;imax>=0;imax--){
        if(hist.at<float>(imax)>minValue)
            break;
    }
    int dim(256);
    Mat lookup(1,&dim,CV_8U);
    for(int i=0;i<256;i++){
        if(i<imin)
            lookup.at<uchar>(i) = 0;
        else if(i>imax)
            lookup.at<uchar>(i) = 255;
        else
            lookup.at<uchar>(i) = static_cast<uchar>(255.0*(i-imin)/(imax-imin)+0.5);
    }
    Mat result;
    result = applyLookUp(image,lookup);
    return result;
}
