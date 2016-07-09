
#include <iostream>
#include <vector>

#include <opencv.hpp>
#include <opencv2/core/core.hpp>

//#define _CRT_SECURE_NO_DEPRECATE//关闭警告

#include "facedetect-dll.h"
#pragma comment(lib, "libfacedetect.lib")

#include "cpreprocess.h"
#include "iofile.h"

using namespace cv;
using namespace std;

CPreprocess::CPreprocess(){}

bool CPreprocess::doit( Mat *imgSrc )
{
	return _do(imgSrc);
}

bool CPreprocess::_do( Mat *imgSrc)
{
	//添加函数
	Mat *img(imgSrc);
	return _detectObjectsCustom(*img);
}
bool CPreprocess::_detectObjectsCustom(Mat &img)
{
	Mat gray(img);
	
	int *pResults = NULL;
	pResults = facedetect_multiview_reinforce((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step, 1.08f, 2, 16);

    // 检测到人脸
	if (*pResults != 0)
	{
		short *p = ((short*)(pResults + 1));
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		//int neighbors = p[4];
		//int angle = p[5];

		if (h < 10 || w < 10)
		{
            return false;
		}
		else
		{
			img = img(Rect(x, y, h, w));
		}
	}
    return true;
}
