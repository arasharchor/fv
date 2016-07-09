#include "cmodelSVM.h"
#include <assert.h>
#include "common.h"

using namespace std;
using namespace cv;

float svm_predict( CvSVM *svm, const Mat& _sample, bool returnDFVal );

	float C;
	int type;
	int max_iter;
	double epsilon;

CModelSVM::CModelSVM(float _C, int _type, int _max_iter, double _epsilon):
			SVM(NULL),C(_C),type(_type),max_iter(_max_iter),epsilon(_epsilon)
{
	SVM = new CvSVM();
}

CModelSVM::~CModelSVM()
{
	if(SVM)
	{
		delete SVM;
		SVM = NULL;
	}
}

void CModelSVM::train(const std::vector<CFeature> &feaSet, const std::vector<float> &labSet)
{
	//---------------------------------0. set train data and labels------------------------------
	assert( feaSet.size()==labSet.size() && feaSet.size() );

	int trainSize = feaSet.size();
	int featureSize = feaSet[0].mFeatureMode.mixfeat.size();

	Mat trains(trainSize, featureSize, CV_32FC1);
	Mat labels(trainSize, 1, CV_32FC1);


	_loadTrain(trains, feaSet);
	_loadLabel(labels, labSet);

	//--------------------------------  train  ------------------------------------------------
	train(trains, labels);
}

void CModelSVM::train(const cv::Mat &trains, const cv::Mat &labels)
{
	//---------------------------------1. set SVM parameters-------------------------------------
	CvSVMParams params;

	params.svm_type = CvSVM::C_SVC;
	params.C = C;
	params.kernel_type = type;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, max_iter, epsilon);

	//---------------------------------2. train--------------------------------------------------
	SVM->train(trains, labels, Mat(), Mat(), params);
}

double CModelSVM::similarity(const CFeature &feat)
{
	Mat sampleMat(1, feat.mFeatureMode.mixfeat.size(), CV_32FC1);
	for(size_t j=0; j<feat.mFeatureMode.mixfeat.size(); j++)
			sampleMat.at<float>(j) = feat.mFeatureMode.mixfeat.at(j);

	float response = svm_predict(SVM, sampleMat, true);
	return sigmoid(response);
}

//=======================Model IO==========================
void CModelSVM::loadModel(std::string model_file)
{
	this->SVM->load(model_file.c_str());
}

void CModelSVM::saveModel(std::string model_file)
{
	char backups[1024];
	
	sprintf(backups, ".//model//%s_SVM_%f_%d_%d_%d_%f", model_file.c_str(), C, type, max_iter, epsilon);
	
	SVM->save(model_file.c_str());
	SVM->save(backups);

}