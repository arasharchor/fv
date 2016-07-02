#ifndef __CMODEL_SVM_H
#define __CMODEL_SVM_H

#include "cmodelInt.h"

class CvSVM;

class CModelSVM : public CModelInt
{
	/* ctor and de-ctor */
public:
	CModelSVM();
	~CModelSVM();

	/* interface */
public:
	void train( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet ) override;

	double similarity( const CFeature &fea ) override;
	void saveModel(std::string model_file) override;
	void loadModel(std::string model_file) override;

private:
	void _loadTrain( cv::Mat &trainData, const std::vector<CFeature> &feaSet );
	void _loadLabel( cv::Mat &labelData, const std::vector<float> &labSet );

	/* member var */
private:
	CvSVM *SVM;
};

#endif