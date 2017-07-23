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

//#define READ_IDEAL_PATH
#define WRITE_CAMERA_PATH

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
    //char fn[512]="VID_20170701_100826";
    //char fn[512]="VID_20170609_231526";
    char fn[512]="VID_20170722_201947";
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
    
    strcpy(path_,path);
    VideoCapture cap(strcat(strcat(path_,fn),video));
    printf("%s\n",path_);
    
    Mat cur, cur2;
    Mat canvas;

    double fps = cap.get(CV_CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
    
    int k=0, k_gyro=0, max_frames = 100;
    
    Mat Tsm(3,3,CV_64F);
    Tsm.at<double>(0,0) = 1; Tsm.at<double>(0,1) = 0; Tsm.at<double>(0,2) = 0;
    Tsm.at<double>(1,0) = 0; Tsm.at<double>(1,1) = 1; Tsm.at<double>(1,2) = 0;
    Tsm.at<double>(2,0) = 0; Tsm.at<double>(2,1) = 0; Tsm.at<double>(2,2) = 1;

    
#ifdef READ_IDEAL_PATH
    // camera ideal path
    char ipath[256]="path_ideal.csv";
    strcpy(path_,path);
    FILE *ip=fopen(strcat(strcat(path_,fn),ipath),"r");
    printf("%s\n",path_);
//    fscanf(cp,"%s\n",buf );
//    printf("%s\n",buf);
    
    vector<double> x_ideal,y_ideal,t_ideal;
    while(EOF!=fscanf(ip,"%lf, %lf, %lf\n",&time,&gx,&gy)) {
        t_ideal.push_back(time);
        x_ideal.push_back(gx);
        y_ideal.push_back(gy);
        //        printf("%lf, %lf, %lf, %lf\n",gx,gy,gz,time);
    }
    fclose(ip);
#endif
    

#ifdef WRITE_CAMERA_PATH
    char cpath[256]="path.csv";
    FILE *cp=fopen(strcat(strcat(path_,fn),cpath),"w");
#endif
    
    
    
    vector<double> mat1,mat2,mat3,mat4,mat5,mat6,mat7,mat8,mat9;
    vector<double> mat1s,mat2s,mat3s,mat4s,mat5s,mat6s,mat7s,mat8s,mat9s;
    vector<double> mat_time;
    
    while(k < max_frames-1) { // don't process the very last frame, no valid transform
        
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

        float rec=0.99;
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
        
//        −0.1 ≤ bt, ct ≤ 0.1, −0.05 ≤ bc + ct ≤ 0.05, and
//        −0.1 ≤ at − dt ≤ 0.1.
        
//        warpPerspective(cur, cur2, T2, cur.size());
        
        printf("%0.3lf, %0.3lf, %0.3lf\n",T2d.at<double>(0,0),T2d.at<double>(0,1),T2d.at<double>(0,2));
        printf("%0.3lf, %0.3lf, %0.3lf\n",T2d.at<double>(1,0),T2d.at<double>(1,1),T2d.at<double>(1,2));
        printf("%0.3lf, %0.3lf, %0.3lf\n",T2d.at<double>(2,0),T2d.at<double>(2,1),T2d.at<double>(2,2));


#ifdef WRITE_CAMERA_PATH
        
//        fprintf(cp,"%0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf\n",
//                time,
//                T2d.at<double>(0,0),T2d.at<double>(0,1),T2d.at<double>(0,2),
//                T2d.at<double>(1,0),T2d.at<double>(1,1),T2d.at<double>(1,2),
//                T2d.at<double>(2,0),T2d.at<double>(2,1),T2d.at<double>(2,2));
#endif
        
#ifdef READ_IDEAL_PATH
        //compensation from ideal path
//        x_ideal[k];
        T2d.at<double>(0,2) = T2d.at<double>(0,2) - x_ideal[k];
        T2d.at<double>(1,2) = T2d.at<double>(1,2) - 0;//y_ideal[k];
#endif
        
        
        mat1.push_back(T2d.at<double>(0,0));
        mat2.push_back(T2d.at<double>(0,1));
        mat3.push_back(T2d.at<double>(0,2));
        mat4.push_back(T2d.at<double>(1,0));
        mat5.push_back(T2d.at<double>(1,1));
        mat6.push_back(T2d.at<double>(1,2));
        mat7.push_back(T2d.at<double>(2,0));
        mat8.push_back(T2d.at<double>(2,1));
        mat9.push_back(T2d.at<double>(2,2));

        mat1s.push_back(T2d.at<double>(0,0));
        mat2s.push_back(T2d.at<double>(0,1));
        mat3s.push_back(T2d.at<double>(0,2));
        mat4s.push_back(T2d.at<double>(1,0));
        mat5s.push_back(T2d.at<double>(1,1));
        mat6s.push_back(T2d.at<double>(1,2));
        mat7s.push_back(T2d.at<double>(2,0));
        mat8s.push_back(T2d.at<double>(2,1));
        mat9s.push_back(T2d.at<double>(2,2));
        
        mat_time.push_back(time);
/*
        warpPerspective(cur, cur2, T2d, cur.size());

        //-------------------------------------------------------------------------------------------
        
        //video writer
        if(k==0){
            canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());
            strcpy(path_,path);
            writer.open(strcat(strcat(path_,fn),fndst), CV_FOURCC_DEFAULT, 12, Size(canvas.cols, canvas.rows));
            printf("%s\n",path_);
        }
        cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));
    
//        imshow("after", cur2);
        imshow("after", canvas);


        
        //video writer
//        writer << cur2;
        writer << canvas;
*/
        
        waitKey(10);
        k++;
    }
    puts(".");

#ifdef WRITE_CAMERA_PATH
//    fclose(cp);
#endif
    
    //------------------------------------------------------------------


    
    double th=0.05;//0.1
    double th3=50;
    double th6=50;
    
    
    //3
    
    int init=0;
    for(k = 0; k< max_frames-1; k++) {
        
        if(mat3[k]<mat3[init]-th3 || mat3[init]+th3 < mat3[k]){
            for(int i = init; i < k; i++) {
                mat3s[i]=((mat3[k]-mat3[init])*(i-init))/(k-init) + mat3[init];
            }
            init=k;
        }
    }
    k=max_frames-2;
    for(int i = init; i < k; i++) {
        mat3s[i]=((mat3[k]-mat3[init])*(i-init))/(k-init) + mat3[init];
    }

    //6
    
    init=0;
    for(k = 0; k< max_frames-1; k++) {
        
        if(mat6[k]<mat6[init]-th6 || mat6[init]+th6 < mat6[k]){
            for(int i = init; i < k; i++) {
                mat6s[i]=((mat6[k]-mat6[init])*(i-init))/(k-init) + mat6[init];
            }
            init=k;
        }
    }
    k=max_frames-2;
    for(int i = init; i < k; i++) {
        mat6s[i]=((mat6[k]-mat6[init])*(i-init))/(k-init) + mat6[init];
    }
    
    
    // 1

    init=0;
    for(k = 0; k< max_frames-1; k++) {
        
        if(mat1[k]<mat1[init]-th || mat1[init]+th < mat1[k]){
            for(int i = init; i < k; i++) {
                mat1s[i]=((mat1[k]-mat1[init])*(i-init))/(k-init) + mat1[init];
            }
            init=k;
        }
    }
    k=max_frames-2;
    for(int i = init; i < k; i++) {
        mat1s[i]=((mat1[k]-mat1[init])*(i-init))/(k-init) + mat1[init];
    }

    

    // 2
    
    init=0;
    for(k = 0; k< max_frames-1; k++) {
        
        if(mat2[k]<mat2[init]-th || mat2[init]+th < mat2[k]){
            for(int i = init; i < k; i++) {
                mat2s[i]=((mat2[k]-mat2[init])*(i-init))/(k-init) + mat2[init];
            }
            init=k;
        }
    }
    k=max_frames-2;
    for(int i = init; i < k; i++) {
        mat2s[i]=((mat2[k]-mat2[init])*(i-init))/(k-init) + mat2[init];
    }

    
    // 4
    
    init=0;
    for(k = 0; k< max_frames-1; k++) {
        
        if(mat4[k]<mat4[init]-th || mat4[init]+th < mat4[k]){
            for(int i = init; i < k; i++) {
                mat4s[i]=((mat4[k]-mat4[init])*(i-init))/(k-init) + mat4[init];
            }
            init=k;
        }
    }
    k=max_frames-2;
    for(int i = init; i < k; i++) {
        mat4s[i]=((mat4[k]-mat4[init])*(i-init))/(k-init) + mat4[init];
    }

    
    // 5
    
    init=0;
    for(k = 0; k< max_frames-1; k++) {
        
        if(mat5[k]<mat5[init]-th || mat5[init]+th < mat5[k]){
            for(int i = init; i < k; i++) {
                mat5s[i]=((mat5[k]-mat5[init])*(i-init))/(k-init) + mat5[init];
            }
            init=k;
        }
    }
    k=max_frames-2;
    for(int i = init; i < k; i++) {
        mat5s[i]=((mat5[k]-mat5[init])*(i-init))/(k-init) + mat5[init];
    }

    
    //------------------------------------------------------------------
    strcpy(path_,path);
    VideoCapture cap2(strcat(strcat(path_,fn),video));
    
    printf("%s\n",path_);

    
    fps = cap.get(CV_CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
    
    k=0;
    k_gyro=0;
    max_frames = 100;
    
    
    while(k < max_frames-1) { // don't process the very last frame, no valid transform
        
        cap2 >> cur;
        
        printf("%d\n",k);
        
        if(cur.data == NULL) break;
        
        Mat T2d(3,3,CV_64F);
//        T2d.at<double>(0,0) = mat1[k];
//        T2d.at<double>(0,1) = mat2[k];
//        T2d.at<double>(0,2) = mat3[k];
//        T2d.at<double>(1,0) = mat4[k];
//        T2d.at<double>(1,1) = mat5[k];
//        T2d.at<double>(1,2) = mat6[k];
//        T2d.at<double>(2,0) = mat7[k];
//        T2d.at<double>(2,1) = mat8[k];
//        T2d.at<double>(2,2) = mat9[k];
        T2d.at<double>(0,0) = mat1[k]-mat1s[k]+1.0;
        T2d.at<double>(0,1) = mat2[k]-mat2s[k];
//        T2d.at<double>(0,0) = 1.0;
//        T2d.at<double>(0,1) = 0;
        T2d.at<double>(0,2) = mat3[k]-mat3s[k];
        T2d.at<double>(1,0) = mat4[k]-mat4s[k];
        T2d.at<double>(1,1) = mat5[k]-mat5s[k]+1.0;
//        T2d.at<double>(1,0) = 0;
//        T2d.at<double>(1,1) = 1.0;
        T2d.at<double>(1,2) = mat6[k]-mat6s[k];
//        T2d.at<double>(2,0) = mat7[k];
//        T2d.at<double>(2,1) = mat8[k];
//        T2d.at<double>(2,2) = mat9[k];
        T2d.at<double>(2,0) = 0;
        T2d.at<double>(2,1) = 0;
        T2d.at<double>(2,2) = 1;
        
        
#ifdef WRITE_CAMERA_PATH
        
        fprintf(cp,"%0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf, %0.3lf\n",
                mat_time[k],
                mat1s[k],mat2s[k],mat3s[k],
                mat4s[k],mat5s[k],mat6s[k],
                T2d.at<double>(2,0),T2d.at<double>(2,1),T2d.at<double>(2,2));
        
//                T2d.at<double>(0,0),T2d.at<double>(0,1),T2d.at<double>(0,2),
//                T2d.at<double>(1,0),T2d.at<double>(1,1),T2d.at<double>(1,2),
//                T2d.at<double>(2,0),T2d.at<double>(2,1),T2d.at<double>(2,2));
#endif

        
        
        warpPerspective(cur, cur2, T2d, cur.size());
        
        //-------------------------------------------------------------------------------------------
        
        //video writer
        if(k==0){
            canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());
            strcpy(path_,path);
            writer.open(strcat(strcat(path_,fn),fndst), CV_FOURCC_DEFAULT, 12, Size(canvas.cols, canvas.rows));
            printf("%s\n",path_);
        }
        cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));
        
        //        imshow("after", cur2);
        imshow("after", canvas);
        
        
        
        //video writer
        //        writer << cur2;
        writer << canvas;
        
        
        waitKey(10);
        k++;
    }
#ifdef WRITE_CAMERA_PATH
    fclose(cp);
#endif

    
    
    return 0;
}
