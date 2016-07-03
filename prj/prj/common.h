#ifndef __COMMON_H
#define __COMMON_H

#include <iostream>
#include <opencv.hpp>
#include <string>
#include <math.h>
#include <vector>

std::string itos(int i);

inline double sigmoid(double z){return 1.0/(1+exp(-z));}
void CalFPR_TPR(float &FPR, float &TPR, const std::vector<float> &labelSet, const std::vector<float> &similSet, const float hold);

#endif