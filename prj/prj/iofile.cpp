#include "iofile.h"

using namespace std;


iofile::iofile(string dataListFile)
{
	dataList = dataListFile;
	posCoupleSize = 0;
	negCoupleSize = 0;

	string str;
	ifstream listFile(dataList);

	getline(listFile, str);
	lineLength = str.length();

	for (size_t i = 0; i < str.length(); ++i)
	{
		if (str[i] == ':')
		{
			subStringPos.push_back(i);
		}
	}

	if (str[subStringPos[1] + 1] == '1')			// label
	{
		posCoupleSize++;
	}
	else
	{
		negCoupleSize++;
	}

	while(!listFile.eof())
	{
		getline(listFile, str);
		if (str.length() == lineLength)
		{
			if (str[subStringPos[1] + 1] == '1')	// label
			{
				posCoupleSize++;
			}
			else
			{
				negCoupleSize++;
			}	
		}
	}

	listFile.close();
}

int iofile::posCoupleNums()
{
	return posCoupleSize;
}

int iofile::negCoupleNums()
{
	return negCoupleSize;
}

void iofile::extCoupleImageInf(coupleImageInf &inf, int nth)
{
	ifstream fp(dataList);
	fp.seekg(nth * (lineLength + 2));

	string str;
	getline(fp, str);

	inf.imgPath1 = str.substr(0, subStringPos[0]);
	inf.imgPath2 = str.substr(subStringPos[0] + 1, subStringPos[0]);
	inf.label = str.back() - '0';
}

bool iofile::readFeature(vector<double> &feat, int label, int nth)
{
	fstream fp("Dataset.feat");

	string lineStr;
	getline(fp, lineStr);
    if (lineStr.empty())
    {
        return false;
    }

	fp.seekg(nth * (lineStr.length() + 2));
	getline(fp, lineStr);
	if (lineStr.empty())
	{
		return false;
	}

	istringstream is(lineStr);
	double data;
	while(is >> data)
	{
		feat.push_back(data);
	}
    label = (int)feat.back();
    feat.pop_back();
	fp.close();
	return true;
}

void iofile::writeFeature(vector<double> &feat, int label, int nth)
{
    ifstream fp("Dataset.feat");
    string lineStr;
    getline(fp, lineStr);
    fp.close();

    ofstream tp("Dataset.feat", ios::app | ios::beg);
    tp.seekp(nth* (lineStr.length() + 2));
    for (size_t i = 0; i < feat.size(); ++i)
    {
        tp << setprecision(6) << scientific << feat[i] << " ";
    }
    tp << label << endl;
    tp.close();
}

void iofile::writeErrorLog(const errLogInf &errLog)
{
	ofstream fp("errInf.log", ios::app);	// 'w+'

	fp << errLog.errInf.imgPath1
		<< ':'
		<< errLog.errInf.imgPath2
		<< ':'
		<< errLog.errInf.label
		<< '@'
        << errLog.errImg
		<< '&'
        << setprecision(4) << scientific << (double)errLog.errOrder
		<< endl;
	fp.close();
}

void iofile::readErrorLog(errLogInf &errLog, int nth)
{
    ifstream fp("errInf.log");
    string lineStr;
    getline(fp, lineStr);

    fp.seekg(ios::beg);
    fp.seekg(nth * (lineStr.length() + 2));
    getline(fp, lineStr);

    errLog.errInf.imgPath1 = lineStr.substr(0, subStringPos[0]);
    errLog.errInf.imgPath2 = lineStr.substr(subStringPos[0] + 1, subStringPos[0]);
    errLog.errInf.label = lineStr[subStringPos[1] + 1] - '0';
    errLog.errImg = lineStr[subStringPos[1] + 3] - '0';
    
    lineStr = lineStr.substr(subStringPos[1] + 5, lineStr.length() - subStringPos[1] - 5);

    double nums;
    istringstream is(lineStr);
    is >> nums;
    errLog.errOrder = nums;
    fp.close();
}

void iofile::outputSimilarFile(const vector<float> similSet)
{
    ofstream fp("хавЃ_Distance.txt");
    fp << setiosflags(ios::fixed) << setprecision(6) << 0.5 << endl;
    for (size_t i = 0; i < similSet.size(); i++)
    {
        fp << setiosflags(ios::fixed) << setprecision(6) << similSet[i] << endl;
    }
    fp.close();
}

void iofile::outputRocFile(const vector<pair<float,float> > rocSet)
{
    ofstream fp("хавЃ_ROC.txt");
    for (size_t i = 0; i < rocSet.size(); i++)
    {
        fp << setiosflags(ios::fixed) << setprecision(4)
            << 0.01 * i << " " 
            << rocSet[i].second << " "
            << rocSet[i].first << endl;
    }
    fp.close();
}