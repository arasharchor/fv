#include "cmodelSVM.h"
#include "cfeature.h"

using namespace std;
using namespace cv;

CModelSVMStore::~CModelSVMStore()
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

	//---------------------------------1. set SVM parameters-------------------------------------
	CvSVMParams params;

	params.svm_type = CvSVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1e4, 1e-6);

	//---------------------------------2. train--------------------------------------------------
	mModelStore.SVM->train(trains, labels, Mat(), Mat(), params);
}

void CModelSVM::validation_model( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet )
{
	assert( feaSet.size()==labSet.size() && feaSet.size() );

	float accuracy=0;

	cout<<"respone\tlabel"<<endl;
	for(int i=0; i<feaSet.size(); i++)
	{
		Mat sampleMat(1, feaSet[i].mFeatureMode.mixfeat.size(), CV_32FC1);
		
		for(int j=0; j<feaSet[i].mFeatureMode.mixfeat.size(); j++)
			sampleMat.at<float>(j) = feaSet[i].mFeatureMode.mixfeat.at(j);

		float response = mModelStore.SVM->predict(sampleMat);
		cout<<i<<" . "<<response<<"\t"<<labSet[i]<<endl;
		accuracy += ((response==labSet[i])?1:0);
	}
	cout<<"accuracy : "<<accuracy/labSet.size()*100<<"%"<<endl;
}

double CModelSVM::similarity(const CFeature &feat)
{
	return 0.0;
}


//=======================Model IO==========================
void CModelSVM::loadModel(std::string model_file)
{
	this->mModelStore.SVM->load(model_file.c_str());
}

void CModelSVM::saveModel(std::string model_file)
{
	this->mModelStore.SVM->save(model_file.c_str());
}

//=======================load sample========================
void CModelSVM::_loadTrain( cv::Mat &trainData, const std::vector<CFeature> &feaSet )
{
	for(int i=0; i<feaSet.size(); i++)
	{
		for(int j=0; j<feaSet[i].mFeatureMode.mixfeat.size(); j++)
		{
			trainData.at<float>(i, j) = feaSet[i].mFeatureMode.mixfeat[j];
		}
	}
}

void CModelSVM::_loadLabel( cv::Mat &labelData, const std::vector<float> &labSet )
{
	for(int i=0; i<labSet.size(); i++)
	{
		labelData.at<float>(i) = labSet.at(i);
	}
}

CModelSVMStore::CModelSVMStore():SVM(NULL)
{
	SVM = new CvSVM();
}