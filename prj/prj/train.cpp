
#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>


#include "common.h"
#include "cmodelSVM.h"
#include "cmodelANN.h"
#include "cfeature.h"
#include "iofile.h"

static int trainNums = 8000;			// ѵ��������
static int jumpNums = 0;               // ����

using namespace std;
using namespace cv;

void train(void)
{
    iofile obj("datalist.FERET",            // ���ݼ�
                "Dataset.FERET",             // ������
                "��ң_Distance.txt",         // ���ƶ�
                "��ң_ROC.txt",              // ROC
                "errInf.log"                // ��־
                );

	Mat labelSet = Mat::zeros(trainNums, 1, CV_32FC1);                           // ��ǩ��

    // 1).��ȡ����ͼ�������
    for(int i = 0; i < trainNums / 2; i++)
    {
		labelSet.at<float>(i) = 1.0;
    }
    for(int i = 0; i < trainNums / 2; i++)
    {
		labelSet.at<float>(i + trainNums / 2) = -1.0;
    }
	
    Mat featureSet;                                 //������
    obj.load(featureSet, trainNums, jumpNums);      // ����
    //2).ѵ��ģ��
//    CModelInt *model = new CModelSVM(0.00001, CvSVM::LINEAR);
//    model->train(featureSet, labelSet);
//    model->saveModel("model");
	CModelInt *model = new CModelANN(0.001, 5, CvANN_MLP::GAUSSIAN, 5e2);
	model->train(featureSet, labelSet);
	model->saveModel("model");
//	delete model;
}