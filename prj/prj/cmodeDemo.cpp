
#include <fstream>
#include "cmodelDemo.h"
#include "cfeature.h"

using namespace std;

void CModelDemo::train( const vector<CFeature> &feaSet )
{
	mModelStore.param.clear();
	mModelStore.param.resize(2, 10);
}

double CModelDemo::similarity( const CFeature &fea )
{
	return 0;
}

void CModelDemo::saveModel( string model_file )
{
	ofstream ofs(model_file);
	for(int i=0; i<mModelStore.param.size(); i++)
	{
		ofs<<mModelStore.param[i]<<endl;
	}
	ofs.close();
}

void CModelDemo::loadModel( string model_file )
{
	ifstream ifs(model_file);
	string strBuf;
	mModelStore.param.clear();
	while( ifs>>strBuf )
	{
		double value = atof(strBuf.c_str());
		mModelStore.param.push_back(value);
	}
	ifs.close();
}