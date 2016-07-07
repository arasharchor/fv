#ifndef __CMODEL_ANN_H
#define __CMODEL_ANN_H

#include "cmodelInt.h"

class CvANN_MLP;

class CModelANN : public CModelInt
{
	/* ctor and de-ctor */
public:
	CModelANN();
	~CModelANN();
	/* interface */
public:
	void train( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet ) override;
	void validation_model( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet ) override;
	double similarity( const CFeature &fea ) override;
	void saveModel(std::string model_file) override;
	void loadModel(std::string model_file) override;

	/* member var */
private:
	CvANN_MLP *ANN;
};

#endif