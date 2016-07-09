
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

static int trainNums = 7200;			// ѵ��������
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

    Mat labelSet;                           // ��ǩ��

    // 1).��ȡ����ͼ�������
    for(int i = 0; i < trainNums / 2; i++)
    {
        labelSet.push_back(1.0);
    }
    for(int i = 0; i < trainNums / 2; i++)
    {
        labelSet.push_back(-1.0);
    }
    //labelSet = labelSet.t();              // ��������>������

    Mat featureSet;                                 //������
    obj.load(featureSet, trainNums, jumpNums);      // ����

    //2).ѵ��ģ��
    //CModelInt *model = new CModelSVM();
    //model->train(featureSet, labelSet);
    //model->saveModel("svm_model");
    
	CModelInt *model = new CModelANN(0.001, 10, 2);
	model->train(featureSet, labelSet);
	model->saveModel("ann_model");
	delete model;
}