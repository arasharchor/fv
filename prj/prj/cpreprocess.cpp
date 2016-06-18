
#include <iostream>
#include <opencv.hpp>
#include <opencv2/core/core.hpp>
#include "cpreprocess.h"
#include "wrap.h"
#include <vector>

using namespace cv;
using namespace std;

CPreprocess::CPreprocess()
{
	//_do(imgWrapSrc);
	classifier = new CascadeClassifier("haarcascade_frontalface_alt2.xml");
	scaledWidth = 320;
	flags = CASCADE_FIND_BIGGEST_OBJECT;
	minFeatureSize = Size(20, 20);
	searchScaleFactor = 1.1f;
	minNeighbors = 4;
}

void CPreprocess::doit( ImgWrap *imgWrapSrc )
{
	_do(imgWrapSrc);
}

void CPreprocess::_do( ImgWrap *imgWrapSrc)//void CPreprocess::_do(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
	//��Ӻ���
	Mat *img = (Mat *)imgWrapSrc->context;

//	imshow("img", *img);cvWaitKey(0);
	_detectObjectsCustom(*img, *classifier, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
	_drawFaceImage(*img, objects);
	//_detectLargestObject(*img, *classifier, largestObject, scaledWidth);
	//_drawFaceImage(*img, largestObject);
}
void CPreprocess::_detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
    //����ɫͼ��ת��Ϊ�Ҷ�
    Mat gray;
    if (img.channels() == 3) 
	{
        cvtColor(img, gray, CV_BGR2GRAY);
    }
    else if (img.channels() == 4) 
	{
        cvtColor(img, gray, CV_BGRA2GRAY);
    }
    else
	{
        // �����Ϊ�Ҷ�ͼ��.
        gray = img;
    }

    // ����ͼ���ó������еĸ���.
    Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        // �ڱ�����ͬ�ݺ�ȵ�ͬʱ��Сͼ��.
        int scaledHeight = cvRound(img.rows / scale);
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));
    }
    else {
        // ֱ�ӷ��ʴ������ź��ͼ��.
        inputImg = gray;
    }

    // ��һ��ͼ��ĶԱȶȺ�����.
    Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    // ������ź�ĻҶ�ͼ���е�Ŀ������.
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    //����ڼ��ǰ��ʱ��Сͼ�񣬷Ŵ�Ľ����
    if (img.cols > scaledWidth)
	{
        for (int i = 0; i < (int)objects.size(); i++ )
		{
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // ȷ����Ŀ��������ȫ��ͼ���ڣ��Է����ڱ߽��ϵ��������.
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols)
            objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows)
            objects[i].y = img.rows - objects[i].height;
    }

    // Return with the detected face rectangles stored in "objects".
}

void CPreprocess::_detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    // ֻ���ͼ��������Ŀ������ 
    int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
    //��СĿ�������ĳߴ�.
    Size minFeatureSize = Size(20, 20);
    // �̶���ʽ��Ϊ�˼�⵽�����ϸ�ڲ���Ӧ�ô���1.0
    float searchScaleFactor = 1.1f;
    int minNeighbors = 4;

    // ִ�м�����
    vector<Rect> objects;
    _detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size() > 0) 
	{
        // ����Ψһ�ļ��Ķ���.
        largestObject = (Rect)objects.at(0);
    }
    else
	{
        largestObject = Rect(-1,-1,-1,-1);
    }
}
//����ʵ�ֺ���

void CPreprocess::_drawFaceImage(Mat img, vector<Rect> objects)
{
	rectangle(img, objects[0], Scalar(255,0,0));  
	imshow("face",img);
	waitKey(0);
}
void CPreprocess::_drawFaceImage(Mat img, Rect largestObject)
{
	rectangle(img, largestObject, Scalar(255,0,0));  
	imshow("face",img);
	waitKey(0);
}