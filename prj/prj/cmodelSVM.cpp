#include "cmodelSVM.h"
#include "cfeature.h"
#include <vector>

#include <opencv2\ml\ml.hpp>
#include <opencv2\core\core.hpp>

using namespace std;
using namespace cv;

void CModelSVM::train(const std::vector<CFeature> &feaSet)
{
	//---------------------------------0. set train data and labels------------------------------
	int trainSize = feaSet.size();
	int featureSize = 2;
	Mat trainData(trainSize, featureSize, CV_32FC1);
	Mat labels(trainSize, 1, CV_32FC1);

	//---------------------------------1. set SVM parameters-------------------------------------
	CvSVMParams params;

	params.svm_type = CvSVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, int(1e7), 1e-6);

	//---------------------------------2. train--------------------------------------------------
//	mModelStore.SVM->train(trainData, labels, Mat(), Mat(), params);
}

double CModelSVM::similarity(const CFeature &feat)
{
	return 0.0;
}

void CModelSVM::loadModel(std::string model_file)
{
	this->mModelStore.SVM->load(model_file.c_str());
}

void CModelSVM::saveModel(std::string model_file)
{
	this->mModelStore.SVM->save(model_file.c_str());
}

CModelSVMStore::CModelSVMStore():SVM(NULL)
{
	SVM = new CvSVM();
}

CModelSVMStore::~CModelSVMStore()
{
	if(SVM)
	{
		delete SVM;
		SVM = NULL;
	}
}
