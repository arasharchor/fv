#include "cmodelANN.h"
#include <assert.h>
#include "common.h"

using namespace std;
using namespace cv;

int ann_train( CvANN_MLP *ann, const Mat& _inputs, const Mat& _outputs,
                     const Mat& _sample_weights, const Mat& _sample_idx,
                     CvANN_MLP_TrainParams _params, int flags=0 );

CModelANN::CModelANN():ANN(NULL)
{
	ANN = new CvANN_MLP();
}

CModelANN::~CModelANN()
{
	if(ANN)
	{
		delete ANN;
		ANN = NULL;
	}
}

void CModelANN::train(const std::vector<CFeature> &feaSet, const std::vector<float> &labSet)
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
	CvANN_MLP_TrainParams params;  
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;  
    params.bp_dw_scale=0.001;  
    params.bp_moment_scale=0.01; 
	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1e3, 1e-100 );
	Mat layerSizes=(Mat_<int>(1,3) << featureSize, 10, 1);  
 //   ANN->create(layerSizes,CvANN_MLP::SIGMOID_SYM);//CvANN_MLP::SIGMOID_SYM  
	ANN->create(layerSizes,CvANN_MLP::GAUSSIAN);

	//---------------------------------2. train--------------------------------------------------
	ann_train(this->ANN, trains, labels, Mat(),Mat(), params);  
}

void CModelANN::validation_model( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet )
{
	assert( feaSet.size()==labSet.size() && feaSet.size() );
}

double CModelANN::similarity(const CFeature &feat)
{
	Mat sampleMat(1, feat.mFeatureMode.mixfeat.size(), CV_32FC1);
	for(int j=0; j<feat.mFeatureMode.mixfeat.size(); j++)
			sampleMat.at<float>(j) = feat.mFeatureMode.mixfeat.at(j);
	Mat responseMat;
	ANN->predict(sampleMat, responseMat);
	float response = *responseMat.ptr<float>(0);

	return sigmoid(response);
//	response = response>1.0 ? 1.0 : (response<-1.0 ? -1.0 : response);	//оч╥Ы
//	return  response;
}

//=======================Model IO==========================
void CModelANN::loadModel(std::string model_file)
{
	ANN->load(model_file.c_str());
}

void CModelANN::saveModel(std::string model_file)
{
	ANN->save(model_file.c_str());
}