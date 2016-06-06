#ifndef __CMODEL_DEMO_H
#define __CMODEL_DEMO_H

#include "cmodelInt.h"
#include <vector>

class CModelDemoStore
{
public:
	CModelDemoStore(){ param.clear(); }
	std::vector<double> param;
};

class CModelDemo : public CModelInt
{
	/* interface */
public:
	void train( const std::vector<CFeature> &feaSet ) override;
	double similarity( const CFeature &fea1, const CFeature &fea2 ) override;
	void storeModel(std::string model_file) override;
	void readModel(std::string model_file) override;

	/* member var */
private:
	CModelDemoStore mModelStore;
};

#endif