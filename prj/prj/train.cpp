#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <string>
#include <opencv.hpp>


#include "common.h"
#include "cmodelSVM.h"
#include "cfeature.h"
#include "iofile.h"

static int TRAIN_NUM = 2000;			//ѵ��ͼ��Ը���

using namespace std;
using namespace cv;

void train(void)
{
    iofile imgCoupleDataSet("datalist.txt");

    vector<CFeature> featureSet(TRAIN_NUM);	//������
	vector<float> labelSet(TRAIN_NUM);

    // 1).��ȡ����ͼ�������
    for(int i = 0; i < TRAIN_NUM; i++)
    {
        featureSet[i] = CFeature(imgCoupleDataSet, i);
        printf("finish %d\n", i);
    }

    //2).ѵ��ģ��
    CModelInt *model = new CModelSVM();
    model->train(featureSet, labelSet);
    model->saveModel("svm_model");

}