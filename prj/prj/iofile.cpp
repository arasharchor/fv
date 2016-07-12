#include <opencv.hpp>
#include "iofile.h"

using namespace std;
using namespace cv;

iofile::iofile(string datalist = "datalist.txt", 
                string featlist = "Dataset.feat",
                string distfile = "逍遥_Distance.txt",
                string rocfile = "逍遥_ROC.txt",
                string errfile = "errInf.log"
                )
{
	dataList = datalist;
    featList = featlist;;
    errorLog = errfile;;
    distFile = distfile;;
    rocFile = rocfile;

	posNums = 0;
	negNums = 0;

	string str;
	ifstream fp(dataList);

	getline(fp, str);
	lineLength = str.length();

    colonPos.push_back(str.find(':'));
    colonPos.push_back(str.rfind(':'));

    fp.seekg(ios::beg);
	while(!fp.eof())
	{
		getline(fp, str);
		if (str.length() == lineLength)
		{
            (str.back() == '1') ? (posNums++) : (negNums++);
		}
	}
	fp.close();
}

int iofile::posSamplesNums(void)
{
    return posNums;
}

int iofile::negSamplesNums(void)
{
    return negNums;
}

int iofile::rowNums(void)
{
    return (posNums + negNums);
}

int iofile::cloNums(void)
{
    ifstream fp(featList);
    string str;
    int cnt = 0;
    double data;
    getline(fp, str);
    istringstream is(str);
    while(is >> data)
    {
        cnt++;
    }
    fp.close();
    return cnt;
}

// 读取第n个样本图相对的路径信息
void iofile::load(coupleImageInf &inf, int nth)
{
	ifstream fp(dataList);
	fp.seekg(nth * (lineLength + 2));

	string str;
	getline(fp, str);
    fp.close();

	inf.imgPath1 = str.substr(0, colonPos[0]);
	inf.imgPath2 = str.substr(colonPos[0] + 1, colonPos[0]);
}

// 读取第n个样本的特征
bool iofile::load(vector<double> &feat, int nth)
{
	fstream fp(featList);

	string str;
	getline(fp, str);
    if (str.empty())
    {
        return false;
    }

	fp.seekg(nth * (str.length() + 2));
	getline(fp, str);
	if (str.empty())
	{
		return false;
	}

	istringstream is(str);
	double data;
	while(is >> data)
	{
		feat.push_back(data);
	}
	fp.close();
	return true;
}

// 保存第n个样本的特征
void iofile::save(vector<double> &feat, int nth)
{
    ifstream fp(featList);
    string str;
    getline(fp, str);
    fp.close();

    ofstream tp(featList, ios::app | ios::beg);
    tp.seekp(nth* (str.length() + 2));
    for (size_t i = 0; i < feat.size() - 1; ++i)
    {
        tp << setprecision(6) << scientific << feat[i] << " ";
    }
    tp << setprecision(6) << scientific << feat[feat.size() - 1] << endl;
    tp.close();
}

void iofile::save(const Mat &featureSet)
{
    ofstream fp(featList);
    fp.seekp(ios::trunc);
    fp.close();
    ofstream fs(featList);
    for (int i = 0; i < featureSet.rows; i++)
    {
        for (int j = 0; j < featureSet.cols; j++)
        {
            fs << setprecision(6) << scientific << featureSet.at<float>(i,j) << " ";
        }
        fs << endl;
    }
    fs.close();
}

// 输出到错误日志
void iofile::save(const errLogInf &errLog)
{
	ofstream fp(errorLog, ios::app);	// 'w+'

	fp << errLog.errInf.imgPath1
		<< ':'
		<< errLog.errInf.imgPath2
		<< ':'
		<< errLog.errInf.label
		<< '@'
        << errLog.errImg
		<< '&'
        << setprecision(4) << scientific << (float)errLog.errOrder
		<< endl;

	fp.close();
}

// 读取错误日志
void iofile::load(errLogInf &errLog, int nth)
{
    ifstream fp(errorLog);
    string lineStr;
    getline(fp, lineStr);

    fp.seekg(ios::beg);
    fp.seekg(nth * (lineStr.length() + 2));
    getline(fp, lineStr);

    errLog.errInf.imgPath1 = lineStr.substr(0, colonPos[0]);
    errLog.errInf.imgPath2 = lineStr.substr(colonPos[0] + 1, colonPos[0]);
    errLog.errInf.label = lineStr[colonPos[1] + 1] - '0';
    errLog.errImg = lineStr[colonPos[1] + 3] - '0';
    
    lineStr = lineStr.substr(colonPos[1] + 5, lineStr.length() - colonPos[1] - 5);

    int nums;
    istringstream is(lineStr);
    is >> nums;
    errLog.errOrder = nums;
    fp.close();
}

// 保存距离、相似度数据
void iofile::save(const vector<float> similSet)
{
    ofstream fp(distFile);
    fp.seekp(ios::trunc);   // 销毁
    fp << setiosflags(ios::fixed) << setprecision(6) << 0.5 << endl;
    for (size_t i = 0; i < similSet.size(); i++)
    {
        fp << setiosflags(ios::fixed) << setprecision(6) << similSet[i] << endl;
    }
    fp.close();
}

// 保存ROC数据
void iofile::save(const vector<pair<float,float> > rocSet)
{
    ofstream fp(rocFile);
    fp.seekp(ios::trunc);
    for (size_t i = 0; i < rocSet.size(); i++)
    {
        fp << setiosflags(ios::fixed) << setprecision(4)
            << 0.01 * i << " " 
            << rocSet[i].second << " "
            << rocSet[i].first << endl;
    }
    fp.close();
}

void iofile::load(Mat &featdata)
{
    ifstream fp(featList);
    string str;
    float data;
    vector<vector<float> > swp;
    while(!fp.eof())
    {
        getline(fp, str);
        istringstream is(str);
        vector<float> tmp;
        while(is >> data)
        {
            tmp.push_back(data);
        }
        swp.push_back(tmp);
    }
    swp.pop_back();

    for (size_t i = 0; i < swp.size(); i++)
    {
        Mat tmp(swp[i]);
        tmp = tmp.t();
        featdata.push_back(tmp);
    }
    fp.close();
}

void iofile::load(Mat &featdata, int trainNums, int jumpNums)
{
    ifstream fp(featList);
    string str;
    getline(fp, str);
	istringstream is(str);
	float data;
	int cnt = 0;
	while (is >> data)
	{
		cnt++;
	}
	featdata = Mat::zeros(trainNums, cnt, CV_32FC1);

    fp.seekg(ios::beg);
    fp.seekg(jumpNums * (str.length() + 2));
    
    for (int i = 0; i < trainNums / 2; i++)
    {
		string stri;
        getline(fp, stri);
        istringstream iw(stri);
		int j = 0;
        while(iw >> data)
        {
			featdata.at<float>(i, j) = data;
			j++;
        }
    }

    fp.seekg(ios::beg);
    fp.seekg((jumpNums + this->posNums) * (str.length() + 2));

    for (int i = 0; i < trainNums / 2; i++)
    {
		string stri;
        getline(fp, stri);
        istringstream iw(stri);
		int j = 0;
        while(is >> data)
        {
			featdata.at<float>(i + trainNums / 2, j) = data;
			j++;
        }
    }
    fp.close();
}

void iofile::dataNormalize(Mat &featdata)
{
    for (int i = 0; i < featdata.cols; i++)
    {
        normalize(featdata.col(i), featdata.col(i), 0.000001, 0.999999, NORM_MINMAX);
    }
}
