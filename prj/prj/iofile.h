#ifndef _IOFILE_H
#define _IOFILE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <io.h>
#include <sstream>
#include <iomanip>
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

	iofile(std::string dataListFile);

	//����������������
	int posCoupleNums();
	int negCoupleNums();

	// ��ȡ��n��������Ϣ
	void extCoupleImageInf(coupleImageInf &inf, int nth);

	// ����д��n����������	[true->����	false->������]
	bool readFeature(std::vector<double> &feat, int nth);
	void writeFeature(std::vector<double> &feat, int nth);

	// ���������־
	void writeErrorLog(const errLogInf &errLog);
    void readErrorLog(errLogInf &errLog, int nth);

    // �ļ����
    void outputSimilarFile(const std::vector<float> similSet);
    void outputRocFile(const std::vector<std::pair<float,float> > rocSet);

private:

	int posCoupleSize;					//��������
	int negCoupleSize;					//��������

	int lineLength;						//�б���һ�еĳ���

	std::string dataList;				//�б��ļ���

	std::vector<int> subStringPos;		//��ȡλ��

};

#endif

