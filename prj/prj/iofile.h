#ifndef _IOFILE_H
#define _IOFILE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cv.h>

//����ͼ��
struct coupleImageInf
{
	std::string imgPath1;		// �����ĵ�1��ͼƬ
	std::string imgPath2;		// -------2-----
	int label;					// ������ǩ
};
// ��־���
struct errLogInf
{
    enum{imgNoErr = 0, img1Err, img2Err, imgAllErr};
	int errOrder;				// �������
	int errImg;					// �����ͼ��
	coupleImageInf errInf;		// ����������Ϣ
};

class iofile
{
public:

    iofile::iofile(std::string datalist, 
        std::string featlist,
        std::string distfile,
        std::string rocfile,
        std::string errfile
        );

	// ������������
	int posSamplesNums(void);
	int negSamplesNums(void);

    int rowNums(void);
    int cloNums(void);
    // ���ݹ�һ��
    void dataNormalize(cv::Mat &featdata);

	// ��ȡ��n������·����Ϣ
	void load(coupleImageInf &inf, int nth);

	// ����д��n����������	[true->����	false->������]
	bool load(std::vector<double> &feat, int nth);
	void save(std::vector<double> &feat, int nth);

    // ������������
    void load(cv::Mat &featdata);
    void load(cv::Mat &featdata, int trainNums, int jumpNums);

	// ���������־
	void save(const errLogInf &errLog);
    void load(errLogInf &errLog, int nth);

    // �ļ����
    void save(const std::vector<float> similSet);
    void save(const std::vector<std::pair<float,float> > rocSet);
    void save(const cv::Mat &featureSet);

private:

	int posNums;					    // ��������
	int negNums;					    // ��������

	int lineLength;						// �б���һ�еĳ���
	std::vector<int> colonPos;		    // ':'��λ��

	std::string dataList;				// ���ݼ��ļ�
    std::string featList;               // �����ļ�
    std::string errorLog;               // ��־�ļ�
    std::string distFile;               // ���롢���ƶ�
    std::string rocFile;                // ROC�ļ�
};

#endif

