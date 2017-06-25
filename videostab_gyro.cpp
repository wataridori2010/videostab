//
//  videostab.cpp
//
//  Created by Ito Yuichi on 2017/06/04.
//  Copyright © 2017年 Ito Yuichi. All rights reserved.
//


#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window



int main(int argc, char **argv)
{
    
    vector<double> timeStamp;
    vector<double> gyrX;
    vector<double> gyrY;
    vector<double> gyrZ;
    
    double time=0,gx = 0,gy=0,gz=0;
    FILE *fp=fopen("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/VID_20170609_230954gyro.csv","r");
    char buf[256];
    fscanf(fp,"%s\n",buf );
    printf("%s\n",buf);
    
    while(EOF!=fscanf(fp,"%lf,%lf,%lf,%lf\n",&gx,&gy,&gz,&time)) {
        timeStamp.push_back(time);
        gyrX.push_back(gx);
        gyrY.push_back(gy);
        gyrZ.push_back(gz);
        printf("%lf, %lf, %lf, %lf\n",gx,gy,gz,time);
    }
    
    double *timeStamp2=(double *)calloc(timeStamp.size(),sizeof(double));
    double *rotX=(double *)calloc(timeStamp.size(),sizeof(double));
    double *rotY=(double *)calloc(timeStamp.size(),sizeof(double));
    double *rotZ=(double *)calloc(timeStamp.size(),sizeof(double));

    // copy and initialize
    for(int i=0; i< timeStamp.size(); i++) timeStamp2[i]=timeStamp[i]-timeStamp[0];

/*
    // remove low freqency
    double *gyrX_low=(double *)calloc(timeStamp.size(),sizeof(double));
    double *gyrY_low=(double *)calloc(timeStamp.size(),sizeof(double));
    double *gyrZ_low=(double *)calloc(timeStamp.size(),sizeof(double));
    double gyrX_rec=gyrX[0];
    double gyrY_rec=gyrY[0];
    double gyrZ_rec=gyrZ[0];
    double rec=0.95;
    
    for(int i=0; i< timeStamp.size()-1; i++){
        gyrX_rec = rec * gyrX_rec + (1-rec) * gyrX[i];
        gyrY_rec = rec * gyrY_rec + (1-rec) * gyrY[i];
        gyrZ_rec = rec * gyrZ_rec + (1-rec) * gyrZ[i];
        gyrX_low[i] = gyrX_rec;
        gyrY_low[i] = gyrY_rec;
        gyrZ_low[i] = gyrZ_rec;
        
        // subtract original gyro data
        gyrX[i] = gyrX[i] - gyrX_low[i];
        gyrY[i] = gyrY[i] - gyrY_low[i];
        gyrZ[i] = gyrZ[i] - gyrZ_low[i];
    }
    
    free(gyrX_low);
    free(gyrY_low);
    free(gyrZ_low);
*/
    
    // accumulating
    double denom=1000000000;
    for(int i=1; i< timeStamp.size()-1; i++){
        rotX[i]=rotX[i-1]+((timeStamp2[i]-timeStamp2[i-1])*gyrX[i-1])/denom;
        rotY[i]=rotY[i-1]+((timeStamp2[i]-timeStamp2[i-1])*gyrY[i-1])/denom;
        rotZ[i]=rotZ[i-1]+((timeStamp2[i]-timeStamp2[i-1])*gyrZ[i-1])/denom;
    }
    
    
    Mat T(2,3,CV_64F);
    Mat E(2,3,CV_64F);
    E.at<double>(0,0) = 1;
    E.at<double>(0,1) = 0;
    E.at<double>(1,0) = 0;
    E.at<double>(1,1) = 1;
    E.at<double>(0,2) = 0;
    E.at<double>(1,2) = 0;
    
    VideoWriter writer;//("test1.avi", CV_FOURCC_DEFAULT, 10, Size(640, 480), true);
    
    VideoCapture cap("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/VID_20170609_230954.mp4");
    Mat cur, cur2;
    Mat canvas;

    double fps = cap.get(CV_CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
    
    int k=0, max_frames = 100;
    
    //while(k < max_frames-1) { // don't process the very last frame, no valid transform
    while(1) {
        
        cap >> cur;
        
        printf("%d\n",k);
        
        if(cur.data == NULL)             break;
        
        float rx=rotX[3*(k+20)]; // should adjust time stamp!!! and Delay
        float ry=rotY[3*(k+20)];
        float rz=rotZ[3*(k+20)];
        
        float w = 960;//1920;
        float h = 544;//1080;
//        float w = 2*960;//1920;
//        float h = 2*544;//1080;
        
        float fov=1.5358444444;// field of view
        
        float dx=-rx*w/fov;
        float dy=-ry*h/fov;
        float da=-rz;
        
        // Restriction
//        if(dx<-100)      dx=-100;//about 10%
//        else if(dx>100)  dx=100;
//        if(dy<-100)      dy=-100;
//        else if(dy>100)  dy=100;
//        if(da<-0.5)     da=-0.5;
//        else if(da>0.5) da=0.5;

        
        T.at<double>(0,0) = cos(rz);
        T.at<double>(0,1) = -sin(rz);
        T.at<double>(1,0) = sin(rz);
        T.at<double>(1,1) = cos(rz);
        T.at<double>(0,2) = dx;
        T.at<double>(1,2) = dy;

        

        warpAffine(cur, cur2, T, cur.size());
        
        //video writer
        if(k==0){
//            writer.open("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/stabilized_result.avi", CV_FOURCC_DEFAULT, 30, Size(cur.cols, cur.rows));
            canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());
            writer.open("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/stabilized_result2.avi", CV_FOURCC_DEFAULT, 30, Size(canvas.cols, canvas.rows));
            
        }
        cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));
//        if(canvas.cols > 1920) {
//            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
//        }

    
//        imshow("after", cur2);
        imshow("after", canvas);


        
        //video writer
//        writer << cur2;
        writer << canvas;
        
        //{
        //char str[256];
        //sprintf(str, "images/%08d.jpg", k);
        //imwrite(str, canvas);
        //}
        
        waitKey(10);
        k++;
    }
    puts(".");

    
    return 0;
}
