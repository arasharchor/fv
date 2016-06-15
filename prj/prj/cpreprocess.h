#ifndef __CPREPROCESS_H
#define __CPREPROCESS_H
#pragma once

#include "cpreprocessInt.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class CPreprocess : public CPreprocessInt
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	CPreprocess();
	void doit( ImgWrap *imgWrapSrc ) override;

	/* member fun */
private:
	void _do( ImgWrap *imgWrapSrc);

	void _detectObjectsCustom(const  Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors);
	void _detectLargestObject(const  Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);//Ѱ��һ��ͼ�������
	//void _detectManyObjects(const ImgWrap *imgWrapSrc, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);//Ѱ�Ҷ��ͼ�������
	void _drawFaceImage(Mat img, vector<Rect> objects);
	void _drawFaceImage(Mat img, Rect largestObject);
	//��Ӻ�����
	/* member var */
private:
	CascadeClassifier *classifier;
	vector<Rect> objects;
	Rect largestObject;
	int scaledWidth;
	int flags;
	Size minFeatureSize;
	float searchScaleFactor;
	int minNeighbors;
};

#endif