clc;
clear;
close all;

roc_data = importdata('../��ң_ROC.txt');
plot(roc_data(:,3), roc_data(:, 2));