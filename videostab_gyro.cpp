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


//Mat  getRodorigue(rx, ry, rz){
//
//    smallR = Rodrigues(np.array([float(rx), float(ry), float(rz)]))[0]
//    R = np.array([[smallR[0][0], smallR[0][1], smallR[0][2], 0],
//                  [smallR[1][0], smallR[1][1], smallR[1][2], 0],
//                  [smallR[2][0], smallR[2][1], smallR[2][2], 0],
//                  [0,         0,         0,         1]])
//    return R
//
//}

int main(int argc, char **argv)
{
    vector<double> timeStamp;
    vector<double> gyrX;
    vector<double> gyrY;
    vector<double> gyrZ;
    char path[512]="/Users/itoyuichi/github/playground/OpenCV/videostab/EIS/";
    char path_[512]="";
    //char fn[512]="VID_20170609_231526";
    //char fn[512]="VID_20170701_182046";// translation
    //char fn[512]="VID_20170701_182111"; // hand-held
    //char fn[512]="VID_20170701_100752"; //
    char fn[512]="VID_20170701_100826";
    //char fn[512]="VID_20170609_231526";
    
    char gyro[512]="gyro.csv";
    char video[512]=".mp4";
    char fndst[512]="_result.mp4";
    
    
    double time=0,gx = 0,gy=0,gz=0;
//    FILE *fp=fopen("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/VID_20170609_230954gyro.csv","r");
//    FILE *fp=fopen("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/VID_20170609_231526gyro.csv","r");
    strcpy(path_,path);
    FILE *fp=fopen(strcat(strcat(path_,fn),gyro),"r");
    printf("%s\n",path_);
    char buf[256];
    fscanf(fp,"%s\n",buf );
    printf("%s\n",buf);
    
    while(EOF!=fscanf(fp,"%lf,%lf,%lf,%lf\n",&gx,&gy,&gz,&time)) {
        timeStamp.push_back(time);
        gyrX.push_back(gx);
        gyrY.push_back(gy);
        gyrZ.push_back(gz);
//        printf("%lf, %lf, %lf, %lf\n",gx,gy,gz,time);
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
    double rec=0.98;

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
*/
/*
    int sw=60;
    for(int i=0; i< timeStamp.size()-1; i++){
        gyrX_rec = 0;
        gyrY_rec = 0;
        gyrZ_rec = 0;
        float counts=0;
        for(int j=-sw; j<= sw; j++){
            if(i+j>=0 && i+j<timeStamp.size()){
            gyrX_rec = gyrX_rec + gyrX[i+j];
            gyrY_rec = gyrY_rec + gyrY[i+j];
            gyrZ_rec = gyrZ_rec + gyrZ[i+j];
            counts+=1.0;
            }
        }
        gyrX_low[i] = gyrX_rec/counts;
        gyrY_low[i] = gyrY_rec/counts;
        gyrZ_low[i] = gyrZ_rec/counts;
    }
    for(int i=0; i< timeStamp.size()-1; i++){
    
        // subtract original gyro data
        gyrX[i] = gyrX[i] - gyrX_low[i];
        gyrY[i] = gyrY[i] - gyrY_low[i];
        gyrZ[i] = gyrZ[i] - gyrZ_low[i];
    }
*/
    
//    free(gyrX_low);
//    free(gyrY_low);
//    free(gyrZ_low);

    
//    double *rotX_sm=(double *)calloc(timeStamp.size(),sizeof(double));
//    double *rotY_sm=(double *)calloc(timeStamp.size(),sizeof(double));
//    double *rotZ_sm=(double *)calloc(timeStamp.size(),sizeof(double));
    
    // accumulating
    double denom=1000000000;
    for(int i=1; i< timeStamp.size()-1; i++){
//        rotX[i]=rotX[i-1]+((timeStamp2[i]-timeStamp2[i-1])*gyrX[i-1])/denom;
//        rotY[i]=rotY[i-1]+((timeStamp2[i]-timeStamp2[i-1])*gyrY[i-1])/denom;
//        rotZ[i]=rotZ[i-1]+((timeStamp2[i]-timeStamp2[i-1])*gyrZ[i-1])/denom;
//        rotX[i]=rotX[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrX[i-1]+gyrX[i])*0.5)/denom;
//        rotY[i]=rotY[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrY[i-1]+gyrY[i])*0.5)/denom;
//        rotZ[i]=rotZ[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrZ[i-1]+gyrZ[i])*0.5)/denom;
        rotX[i]=rotX[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrX[i-1]+gyrX[i]+gyrX[i+1])*0.333)/denom;
        rotY[i]=rotY[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrY[i-1]+gyrY[i]+gyrY[i+1])*0.333)/denom;
        rotZ[i]=rotZ[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrZ[i-1]+gyrZ[i]+gyrZ[i+1])*0.333)/denom;
//        rotX[i]=rotX[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrX[i-1]+2*gyrX[i]+gyrX[i+1])*0.25)/denom;
//        rotY[i]=rotY[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrY[i-1]+2*gyrY[i]+gyrY[i+1])*0.25)/denom;
//        rotZ[i]=rotZ[i-1]+((timeStamp2[i]-timeStamp2[i-1])*(gyrZ[i-1]+2*gyrZ[i]+gyrZ[i+1])*0.25)/denom;
        printf("timeStamp2: %lf, %lf\n",timeStamp2[i]-timeStamp2[i-1],timeStamp[i]-timeStamp[i-1]);
    }
    for(int i=1; i< timeStamp.size()-1; i++){
        timeStamp[i]/=denom;
        timeStamp2[i]/=denom;
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
    
//    VideoCapture cap("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/VID_20170609_230954.mp4");
//    VideoCapture cap("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/VID_20170609_231526.mp4");
//    VideoCapture cap("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/VID_20170609_231125.mp4");
    strcpy(path_,path);
    VideoCapture cap(strcat(strcat(path_,fn),video));
    printf("%s\n",path_);
    
    Mat cur, cur2;
    Mat canvas;

    double fps = cap.get(CV_CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
    
    int k=0, k_gyro=0, max_frames = 1000;
    
    Mat Tsm(3,3,CV_64F);
    Tsm.at<double>(0,0) = 1; Tsm.at<double>(0,1) = 0; Tsm.at<double>(0,2) = 0;
    Tsm.at<double>(1,0) = 0; Tsm.at<double>(1,1) = 1; Tsm.at<double>(1,2) = 0;
    Tsm.at<double>(2,0) = 0; Tsm.at<double>(2,1) = 0; Tsm.at<double>(2,2) = 1;

    
//    while(k < max_frames-1) { // don't process the very last frame, no valid transform
    while(1) {
        
        cap >> cur;
        
        printf("%d\n",k);
        
        if(cur.data == NULL) break;
        
        double st=cap.get(CV_CAP_PROP_POS_MSEC)/1000.0;
        
        //float time=float(k)/24.9423;
        //while(time>timeStamp2[k_gyro])k_gyro++;
        while(st>timeStamp2[k_gyro])k_gyro++;
        double time=st;
        
        //printf("st: %lf, time:%f\n",st,time);
        //printf("k_gyro: %d\n",k_gyro);
        
//        if(k_gyro>10)k_gyro=k_gyro-10;

        int pa=5;//4;//4;//3
        float rx=rotX[k_gyro+pa]; // should adjust time stamp!!! and Delay
        float ry=rotY[k_gyro+pa];
        float rz=rotZ[k_gyro+pa];
//        double rate=(time-timeStamp2[k_gyro-1])/(timeStamp2[k_gyro]-timeStamp2[k_gyro-1]);
//        float rx=rate*rotX[k_gyro+0]+(1-rate)*rotX[k_gyro-1];
//        float ry=rate*rotY[k_gyro+0]+(1-rate)*rotY[k_gyro-1];
//        float rz=rate*rotZ[k_gyro+0]+(1-rate)*rotZ[k_gyro-1];
        
        
//        float rx=((st-timeStamp2[k_gyro-1])*rotX[k_gyro]+(timeStamp2[k_gyro]-st)*rotX[k_gyro-1])/(timeStamp2[k_gyro]-timeStamp2[k_gyro-1]);
//        float ry=((st-timeStamp2[k_gyro-1])*rotY[k_gyro]+(timeStamp2[k_gyro]-st)*rotX[k_gyro-1])/(timeStamp2[k_gyro]-timeStamp2[k_gyro-1]);
//        float rz=((st-timeStamp2[k_gyro-1])*rotZ[k_gyro]+(timeStamp2[k_gyro]-st)*rotX[k_gyro-1])/(timeStamp2[k_gyro]-timeStamp2[k_gyro-1]);
//        float rx=((st-timeStamp2[k_gyro-1])*rotX[k_gyro]+(timeStamp2[k_gyro]-st)*rotX[k_gyro-1])/(timeStamp2[k_gyro]-timeStamp2[k_gyro-1]);
//        float ry=((st-timeStamp2[k_gyro-1])*rotY[k_gyro]+(timeStamp2[k_gyro]-st)*rotX[k_gyro-1])/(timeStamp2[k_gyro]-timeStamp2[k_gyro-1]);
//        float rz=((st-timeStamp2[k_gyro-1])*rotZ[k_gyro]+(timeStamp2[k_gyro]-st)*rotX[k_gyro-1])/(timeStamp2[k_gyro]-timeStamp2[k_gyro-1]);

        
        
        float w = 960;//1920;
        float h = 544;//1080;
        
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

/*
        T.at<double>(0,0) = cos(da);
        T.at<double>(0,1) = -sin(da);
        T.at<double>(1,0) = sin(da);
        T.at<double>(1,1) = cos(da);
        T.at<double>(0,2) = dx;
        T.at<double>(1,2) = dy;
        
        warpAffine(cur, cur2, T, cur.size());
*/
        //-------------------------------------------------------------------------------------------
        // Rodorigue

        Mat A1(4,3,CV_64F);
        A1.at<double>(0,0) = 1; A1.at<double>(0,1) = 0; A1.at<double>(0,2) = -w/2;
        A1.at<double>(1,0) = 0; A1.at<double>(1,1) = 1; A1.at<double>(1,2) = -h/2;
        A1.at<double>(2,0) = 0; A1.at<double>(2,1) = 0; A1.at<double>(2,2) = 0;
        A1.at<double>(3,0) = 0; A1.at<double>(3,1) = 0; A1.at<double>(3,2) = 1;

        double f=1000;
        //Mat T_(4,4,CV_64F);
        Mat T_(3,4,CV_64F);
        T_.at<double>(0,0) = 1; T_.at<double>(0,1) = 0; T_.at<double>(0,2) = 0;T_.at<double>(0,3) = 0;
        T_.at<double>(1,0) = 0; T_.at<double>(1,1) = 1; T_.at<double>(1,2) = 0;T_.at<double>(1,3) = 0;
        T_.at<double>(2,0) = 0; T_.at<double>(2,1) = 0; T_.at<double>(2,2) = 1;T_.at<double>(2,3) = f;
        //T_.at<double>(3,0) = 0; T_.at<double>(3,1) = 0; T_.at<double>(3,2) = 0;T_.at<double>(0,2) = 0;

        Mat rotation_vector(3,1,CV_64F);
        Mat R(3,3,CV_64F);
        
        double gain=1.0;
        rotation_vector.at<double>(0,0)=-ry*gain;
        rotation_vector.at<double>(1,0)=-rx*gain;
        rotation_vector.at<double>(2,0)=-rz*gain;
        Rodrigues(rotation_vector,R);
        
        
        //Mat A2(3,4,CV_64F);
        Mat A2(3,3,CV_64F);
        A2.at<double>(0,0) = f; A2.at<double>(0,1) = 0; A2.at<double>(0,2) = w/2;//A2.at<double>(0,3) = 0;
        A2.at<double>(1,0) = 0; A2.at<double>(1,1) = f; A2.at<double>(1,2) = h/2;//A2.at<double>(1,3) = 0;
        A2.at<double>(2,0) = 0; A2.at<double>(2,1) = 0; A2.at<double>(2,2) = 1;  //A2.at<double>(2,3) = 0;

        Mat T2(3,3,CV_64F);
        T2 = A2*(R*(T_*A1));
        
        double deno=T2.at<double>(2,2);
        T2.at<double>(0,0) /= deno; T2.at<double>(0,1) /= deno; T2.at<double>(0,2) /= deno;//A2.at<double>(0,3) = 0;
        T2.at<double>(1,0) /= deno; T2.at<double>(1,1) /= deno; T2.at<double>(1,2) /= deno;//A2.at<double>(1,3) = 0;
        T2.at<double>(2,0) /= deno; T2.at<double>(2,1) /= deno; T2.at<double>(2,2) /= deno;  //A2.at<double>(2,3) = 0;

//        printf("%0.3lf, %0.3lf, %0.3lf\n",T2.at<double>(0,0),T2.at<double>(0,1),T2.at<double>(0,2));
//        printf("%0.3lf, %0.3lf, %0.3lf\n",T2.at<double>(1,0),T2.at<double>(1,1),T2.at<double>(1,2));
//        printf("%0.3lf, %0.3lf, %0.3lf\n",T2.at<double>(2,0),T2.at<double>(2,1),T2.at<double>(2,2));

        float rec=0.95;
        Tsm.at<double>(0,0) = rec*Tsm.at<double>(0,0)+(1-rec)*T2.at<double>(0,0);
        Tsm.at<double>(0,1) = rec*Tsm.at<double>(0,1)+(1-rec)*T2.at<double>(0,1);
        Tsm.at<double>(0,2) = rec*Tsm.at<double>(0,2)+(1-rec)*T2.at<double>(0,2);
        Tsm.at<double>(1,0) = rec*Tsm.at<double>(1,0)+(1-rec)*T2.at<double>(1,0);
        Tsm.at<double>(1,1) = rec*Tsm.at<double>(1,1)+(1-rec)*T2.at<double>(1,1);
        Tsm.at<double>(1,2) = rec*Tsm.at<double>(1,2)+(1-rec)*T2.at<double>(1,2);
        Tsm.at<double>(2,0) = rec*Tsm.at<double>(2,0)+(1-rec)*T2.at<double>(2,0);
        Tsm.at<double>(2,1) = rec*Tsm.at<double>(2,1)+(1-rec)*T2.at<double>(2,1);
        Tsm.at<double>(2,2) = rec*Tsm.at<double>(2,2)+(1-rec)*T2.at<double>(2,2);

        Mat T2d(3,3,CV_64F);
        
        T2d=T2*Tsm.inv();
        
        
        if(T2d.at<double>(0,0)< 0.9)    T2d.at<double>(0,0) =   0.9;
        else if(T2d.at<double>(0,0)>1.1)T2d.at<double>(0,0) =   1.1;
        if(T2d.at<double>(1,1)<0.9)     T2d.at<double>(1,1) =   0.9;
        else if(T2d.at<double>(1,1)>1.1)T2d.at<double>(1,1) =   1.1;
        
        if(T2d.at<double>(0,1)< -0.1)    T2d.at<double>(0,1) =   -0.1;
        else if(T2d.at<double>(0,1)>0.1)T2d.at<double>(0,1) =   0.1;

        if(T2d.at<double>(1,0)< -0.1)    T2d.at<double>(1,0) =   -0.1;
        else if(T2d.at<double>(1,0)>0.1)T2d.at<double>(1,0) =   0.1;

        float xth=100;
        if(T2d.at<double>(0,2)< -xth)       T2d.at<double>(0,2) =   -xth;
        else if(T2d.at<double>(0,2)>xth)    T2d.at<double>(0,2) =   xth;

        float yth=60;
        if(T2d.at<double>(1,2)< -yth)       T2d.at<double>(1,2) =   -yth;
        else if(T2d.at<double>(1,2)>yth)    T2d.at<double>(1,2) =   yth;
        
        
//        −0.1 ≤ bt, ct ≤ 0.1, −0.05 ≤ bc + ct ≤ 0.05, and
//        −0.1 ≤ at − dt ≤ 0.1.
        
//        warpPerspective(cur, cur2, T2, cur.size());
        
        printf("%0.3lf, %0.3lf, %0.3lf\n",T2d.at<double>(0,0),T2d.at<double>(0,1),T2d.at<double>(0,2));
        printf("%0.3lf, %0.3lf, %0.3lf\n",T2d.at<double>(1,0),T2d.at<double>(1,1),T2d.at<double>(1,2));
        printf("%0.3lf, %0.3lf, %0.3lf\n",T2d.at<double>(2,0),T2d.at<double>(2,1),T2d.at<double>(2,2));

        warpPerspective(cur, cur2, T2d, cur.size());

        
        //処理領域を設定
        cv::Rect roi(xth, yth, cur2.cols-2*xth, cur2.rows-2*yth);
        //cv::Rect roi(xth, yth, 100, 100);
        //出力の初期化(入力画像を複製)
        cv::Mat cur2_ = cur2.clone();
        
        //ROIの設定
        cv::Mat cur2_roi = cur2_(roi).clone();
        resize( cur2_roi, cur2, cur2.size(), 0, 0, INTER_LANCZOS4);
        //-------------------------------------------------------------------------------------------
        
        
        
        
        
        //video writer
        if(k==0){
//            writer.open("/Users/itoyuichi/github/playground/OpenCV/stab-opencv/stabilized_result.avi", CV_FOURCC_DEFAULT, 30, Size(cur.cols, cur.rows));
            canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());

            
            
            strcpy(path_,path);
            writer.open(strcat(strcat(path_,fn),fndst), CV_FOURCC_DEFAULT, 30, Size(canvas.cols, canvas.rows));
            printf("%s\n",path_);
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
