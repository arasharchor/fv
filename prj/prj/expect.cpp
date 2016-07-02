#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <opencv.hpp>


#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "wrap.h"

#include "iofile.h"

using namespace std;
using namespace cv;

void expect(void)
{
	iofile coupleImgDataSet("datalist.txt");
	int predNum=20;

	vector<CFeature>			featureSet(predNum);	//������
	vector<float>				labelSet(predNum);		//��ǩ��
	vector<float>				similSet(predNum);		//���ƶȼ�
	vector<pair<float,float>>	rocSet(100);			//fpr tpr��

	//1).��ȡ�����������������õ���Ӧ��ǩ
	for(int i=0; i<predNum/2; i++)
	{
		tuple<string, string> path;

		coupleImgDataSet.extCoupleImg_path(path, i, true);						// ��i��������

		Mat img1 = imread(get<0>(path), IMREAD_GRAYSCALE);
		Mat img2 = imread(get<1>(path), IMREAD_GRAYSCALE);

		featureSet[i] =  CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
		labelSet[i] = 1.0;
	}

	for(int i=0; i<predNum/2; i++)
	{
		tuple<string, string> path;

		coupleImgDataSet.extCoupleImg_path(path, i, false);						// ��i��������

		Mat img1 = imread(get<0>(path), IMREAD_GRAYSCALE);
		Mat img2 = imread(get<1>(path), IMREAD_GRAYSCALE);

		featureSet[i+predNum/2] =  CFeature(&ImgWrap(&img1), &ImgWrap(&img2));
		labelSet[i+predNum/2] = -1.0;
	}

	//2).����ʶ��ģ��
	CModelInt *model = new CModelSVM();
	model->loadModel("svm_model");

	//3).�����������������ƶ�
	for(int i=0; i<predNum; i++)
	{
		similSet[i] = model->similarity(featureSet[i]);
	}

	//4).��������hold��Ӧ��FPR TPR
	for(int i=0; i<100; i+=1)
	{
		CalFPR_TPR(rocSet[int(i)].first, rocSet[int(i)].second, labelSet, similSet, i*0.01);
	}

	//5).��������ļ�
	/*
	����˵��:
	  1).similSet�������Թ����������ļ�(��Ϊ������ֵ�Ƕ��٣������Զ�����ô��)
	     �����ֵ������0.5�����ǵ�FPR�Ҹ�TPR����ֵ
	  2).rocSet�������Թ���ROC�ı�
	     rocSet[i]ָ��ֵΪi*0.01ʱ��FPR��TPR��rocSet[i]�Ǹ�pair<float, float>���ݣ�first��FPR��ROC���Ǻ��ᣬsecond��TPR��ROC��������
	     FPR�������Ϊ��������ʶ������ʡ���TPR�������Ϊ��������ʶ����ȷ�ʡ�
	*/
}