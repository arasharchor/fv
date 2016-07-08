#ifndef __CMODEL_SVM_H
#define __CMODEL_SVM_H

#include "cmodelInt.h"

class CvSVM;

class CModelSVM : public CModelInt
{
	/* ctor and de-ctor */
public:
	CModelSVM(float _C=1.0, int _type=CvSVM::RBF, int _max_iter=1e6, double _epsilon=1e-10);
	~CModelSVM();
	/* interface */
public:
	void train( const std::vector<CFeature> &feaSet , const std::vector<float> &labSet ) override;
	void train(const cv::Mat &trains, const cv::Mat &labels) override;			//输入特征集和标签集，训练模型
	double similarity( const CFeature &fea ) override;
	void saveModel(std::string model_file) override;
	void loadModel(std::string model_file) override;

	/* member var */
private:
	CvSVM *SVM;
	float C;
	int type;
	int max_iter;
	double epsilon;
};

#endif