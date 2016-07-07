clc;
clear;
close all;

roc_data = importdata('../хавЃ_ROC.txt');
plot(roc_data(:,3), roc_data(:, 2));