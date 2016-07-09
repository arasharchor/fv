#ifndef __CPREPROCESS_H
#define __CPREPROCESS_H
#pragma once

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cpreprocessInt.h"


class CPreprocess : public CPreprocessInt
{
	/* ctor and de-ctor */
public:

	/* interface */
public:
	CPreprocess();
	bool doit( cv::Mat *imgSrc ) override;

	/* member fun */
private:
	bool _do( cv::Mat *imgSrc);

	bool _detectObjectsCustom(cv::Mat &img);

	//Ìí¼Óº¯Êý£»
	/* member var */
private:

};

#endif