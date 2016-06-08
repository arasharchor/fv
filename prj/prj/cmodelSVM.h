#ifndef __CMODEL_SVM_H
#define __CMODEL_SVM_H

#include "cmodelInt.h"

class CvSVM;

class CModelSVMStore
{
public:
	CModelSVMStore();
	~CModelSVMStore();
	CvSVM *SVM;
};

class CModelSVM : public CModelInt
{
	/* interface */
public:
	void train( const std::vector<CFeature> &feaSet ) override;
	double similarity( const CFeature &fea ) override;
	void saveModel(std::string model_file) override;
	void loadModel(std::string model_file) override;

	/* member var */
private:
	CModelSVMStore mModelStore;
};

#endif