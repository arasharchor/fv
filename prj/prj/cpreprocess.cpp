
#include <iostream>
#include <vector>

#include <opencv.hpp>
#include <opencv2/core/core.hpp>

//#define _CRT_SECURE_NO_DEPRECATE//�رվ���

#include "facedetect-dll.h"
#pragma comment(lib,"libfacedetect.lib")

#include "cpreprocess.h"
#include "iofile.h"

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

bool CPreprocess::doit( Mat *imgSrc )
{
	return _do(imgSrc);
}

bool CPreprocess::_do( Mat *imgSrc)//void CPreprocess::_do(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
	//��Ӻ���
	Mat *img(imgSrc);

	//return _detectObjectsCustom(*img, *classifier, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
	return _detectObjectsCustom(*img);
	//_drawFaceImage(*img, objects);
	//_detectLargestObject(*img, *classifier, largestObject, scaledWidth);
	//_drawFaceImage(*img, largestObject);
}
bool CPreprocess::_detectObjectsCustom(Mat &img)
{
	Mat gray(img);//�������ݵ�ַ
	/*facedetect_multiview*/ //facedetect_frontal
	int * pResults = NULL;
	pResults = facedetect_multiview((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step, 1.4f, 2, 24);

	//for(int i = 0; i < (pResults ? *pResults : 0); i++)
	//{
		short * p = ((short*)(pResults+1))+6*0;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		//cvPoint(x, y), cvPoint(x + h, y + w)
		if (h < 10 || w < 10)
		{
			printf("------------------");
            return false;
		}
		else
		{
			img = img(Rect(x, y, h, w));
		}
		
		return true;
	//}

}
bool CPreprocess::_detectObjectsCustom(Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
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
	if (objects.empty())	// �� faile
	{
		return false;
	}
	img = img(objects[0]);
	return true;
	//else
	//{
	//	writeFeatFile("err.log", 0, );
	//}
    // Return with the detected face rectangles stored in "objects".
}

void CPreprocess::_detectLargestObject(Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    // ֻ���ͼ��������Ŀ������ 
    int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;

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
	img = img(largestObject);
}
//����ʵ�ֺ���

void CPreprocess::_drawFaceImage(const Mat img, vector<Rect> objects)
{
//	rectangle(img, objects[0], Scalar(255,0,0));  
	imshow("face",img);
	waitKey(0);
}
void CPreprocess::_drawFaceImage(const Mat img, Rect largestObject)
{
//	rectangle(img, largestObject, Scalar(255,0,0));  
	imshow("face",img);
	waitKey(0);
}