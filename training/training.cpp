/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: training.cpp
    Last modifed:   22.12.2016 by Leonardo Citraro
    Description:    Training of the classifier using the HOG feature

    =========================================================================

    =========================================================================
*/
#include "HOG.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <functional>
#include <ctime>
#include <iomanip>
#include <math.h>

static int MatTYPE = CV_32FC1;
using TYPE = float;

TYPE compute_mean(std::vector<TYPE> v) {
    return std::accumulate(std::begin(v), std::end(v), 0.0f)/v.size();
}

void feature_mean_variance(const cv::Mat& data, std::vector<TYPE>& mean, std::vector<TYPE>& var) {
    mean.resize(data.cols);
    var.resize(data.cols);
    
    for(size_t col=0; col<data.cols; ++col) {
        std::vector<TYPE> feature(data.rows);
        for(size_t i = 0; i < data.rows; ++i) {
            const TYPE* ptr_row = data.ptr<TYPE>(i);
            feature[i] = ptr_row[col];
        }
        TYPE m = std::accumulate(std::begin(feature), std::end(feature), 0.0)/feature.size();
        mean[col] = m;
        std::vector<TYPE> diff(data.rows);
        std::transform(std::begin(feature), std::end(feature), std::begin(diff), std::bind2nd(std::minus<TYPE>(), m));
        TYPE v = std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), 0.0)/feature.size();
        var[col] = v;
    }
}

template<class T>
void save_vector( const std::string& filename, const std::vector<T>& v ) {
    try {
        std::ofstream f(filename, std::ios::binary);
        unsigned int len = v.size();
        f.write( (char*)&len, sizeof(len) );
        f.write( (const char*)&v[0], len * sizeof(T) );
        f.close();
    } catch(...) {
        throw;
    }
}

int main(int argc, char* argv[]) {
    
    // size of the box that should contain a person
    cv::Size person_size(40,120);
    
    // setting up the HOG
    size_t cellsize = 5;
    size_t blocksize = cellsize*2;
    size_t stride = cellsize;
    size_t binning = 9;
    HOG hog(blocksize, cellsize, stride, binning, HOG::GRADIENT_SIGNED, HOG::BLOCK_NORM::none);
    hog.save("hog.ext");
    
    // matrix of data and labels
    std::vector<std::vector<TYPE>> data;
    std::vector<int> labels;

    // open the subimages of the persons one by one
    for(int i=0; i<560; ++i){
        std::string filename = "../dataset/persons/" + std::to_string(i) + ".jpg";
        try {
            // open and display an image
            cv::Mat image = cv::imread(filename, CV_8U);
            if(image.data) {
                // Retrieve the HOG from the image
                hog.process(image);
                auto hist = hog.retrieve(cv::Rect(0,0,person_size.width,person_size.height));
                
                // check for undefined values
                for(auto& h:hist){
                    if(std::isnan(h) || std::isinf(h)) {
                        std::cerr << "h is nan or inf\n";
                        h = static_cast<TYPE>(0.0);
                    }
                }
                data.push_back(hist);
                labels.push_back(1);
            } else
                std::cerr << "invalid image person\n";
        } catch(...) {
            continue;
        }
    }
    
    std::cout << "Conversion of persons images done!\n";
    
    // same for non-persons subimages
    for(int i=0; i<9120; ++i){
        std::string filename = "../dataset/not_persons/" + std::to_string(i) + ".jpg";
        try {
            // open and display an image
            cv::Mat image = cv::imread(filename, CV_8U);
            if(image.data) {
                // Retrieve the HOG from the image
                hog.process(image);
                auto hist = hog.retrieve(cv::Rect(0,0,person_size.width,person_size.height));
                
                for(auto& h:hist){
                    if(std::isnan(h) || std::isinf(h)) {
                        std::cerr << "h is nan or inf\n";
                        h = static_cast<TYPE>(0.0);
                    }
                }
                data.push_back(hist);
                labels.push_back(-1);
            } else
                std::cerr << "invalid image not_person\n";
        } catch(...) {
            continue;
        }
    }
    
    std::cout << "Conversion of non-persons images done!\n";
    
    std::cout << "mat_data=[" << data.size() << " x " << data[0].size() << "]\n";
    std::cout << "mat_labels=[" << labels.size() << " x " << 1 << "]\n";
    
    // ----------------------------------------------------------------------------
    // Convert the std::vector into cv::Mat
    // ----------------------------------------------------------------------------
    cv::Mat mat_labels(labels,false);
    cv::Mat mat_data(data.size(), data[0].size(), MatTYPE);
    for(size_t i = 0; i < mat_data.rows; ++i) {
        for (size_t j = 0; j < mat_data.cols; ++j) {
            TYPE val = data[i][j];
            if(std::isnan(val) || std::isinf(val))
                std::cerr << "val is inf or nan!\n";
            mat_data.at<TYPE>(i,j) = val;
        }
    }
    
    std::cout << "Data packing done!\n";

    // ----------------------------------------------------------------------------
    // Get mean and variance of all features
    // ----------------------------------------------------------------------------
    std::vector<TYPE> mean, var;
    feature_mean_variance(mat_data, mean, var);
    
    save_vector("mean.ext", mean);
    save_vector("var.ext", var);
    //auto temp = load_vector<TYPE>("mean.ext");
    
    std::cout << "Get mean and variance of the features done!\n";
    
    // ----------------------------------------------------------------------------
    // Normalization zero-mean and unit-variance
    // ----------------------------------------------------------------------------
    for(size_t i = 0; i < mat_data.rows; ++i) {
        for(size_t j = 0; j < mat_data.cols; ++j) {
            mat_data.at<TYPE>(i,j) -= mean[j];
            mat_data.at<TYPE>(i,j) /= var[j];
        }
    }

    // ----------------------------------------------------------------------------
    // Model prep and training
    // ----------------------------------------------------------------------------
    cv::Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::create(mat_data, cv::ml::SampleTypes::ROW_SAMPLE, mat_labels);
    
    cv::Ptr<cv::ml::SVM> clf = cv::ml::SVM::create();
    clf->setType(cv::ml::SVM::C_SVC);
    clf->setKernel(cv::ml::SVM::LINEAR);
    //clf->setDegree(2);
    //clf->setNu(0.5);
    //clf->setC(10);
    //clf->setGamma(100);
    clf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 3000, 1e-12));
    
    std::vector<TYPE> accuracies;
    std::vector<TYPE> sensitivities;
    std::vector<TYPE> specificities;
    std::vector<TYPE> false_positive_rates;
    for(int k=0; k<3; k++) {
        
        // random split
        dataset->setTrainTestSplitRatio(0.7, true);
        
        // training & test samples
        cv::Mat train_idx = dataset->getTrainSampleIdx();
        cv::Mat train_data = cv::ml::TrainData::getSubVector(mat_data, train_idx);//dataset->getTrainSamples();
        cv::Mat train_labels = cv::ml::TrainData::getSubVector(mat_labels, train_idx);//dataset->getTrainResponses();
        
        cv::Mat test_idx = dataset->getTestSampleIdx();
        cv::Mat test_data = cv::ml::TrainData::getSubVector(mat_data, test_idx);//dataset->getTestSamples();
        cv::Mat test_labels = cv::ml::TrainData::getSubVector(mat_labels, test_idx);//;//dataset->getTestResponses();
        
        clf->train(train_data, cv::ml::SampleTypes::ROW_SAMPLE, train_labels);
        
        TYPE TP = 0; // true positive
        TYPE TN = 0; // true negative
        TYPE FP = 0; // false positive
        TYPE FN = 0; // false negative
        for(size_t i = 0; i < test_data.rows; ++i) {
            cv::Mat row(1, test_data.cols, MatTYPE, test_data.ptr<TYPE>(i));
            int prediction = clf->predict(row);
            int label = test_labels.at<int>(i);
            if(prediction == label) {
                if(label == 1) ++TP;
                else ++TN;
            }
            if(prediction != label) {
                if(label == 1) ++FN;
                else ++FP;
            }
        }
        TYPE accuracy = (TP+TN)/(TP+TN+FP+FN);
        TYPE sensitivity = (TP)/(TP+FN);
        TYPE specificity = (TN)/(TN+FP);
        TYPE false_positive_rate = (FP)/(FP+TN);
        
        std::cout   << "round=" << k
                    << " accuracy=" << std::setprecision(4) << std::setw(7) << accuracy
                    << " sensitivity=" << std::setprecision(4) << std::setw(7)  << sensitivity
                    << " specificity=" << std::setprecision(4) << std::setw(7)  << specificity
                    << " false_p_r=" << std::setprecision(4) << std::setw(7)  << false_positive_rate << "\n";
        
        accuracies.push_back(accuracy);
        sensitivities.push_back(sensitivity);
        specificities.push_back(specificity);
        false_positive_rates.push_back(false_positive_rate);
    }
    std::cout   << "\n------------- final ------------------------\n";
    std::cout   << "accuracy=" << std::setprecision(4) << std::setw(7)  << compute_mean(accuracies)
                << " sensitivity=" << std::setprecision(4) << std::setw(7)  << compute_mean(sensitivities)
                << " precision=" << std::setprecision(4) << std::setw(7)  << compute_mean(specificities)
                << " false_p_r=" << std::setprecision(4) << std::setw(7)  << compute_mean(false_positive_rates) << "\n\n";
                
    std::cout << "Validation done!\n";
    
    clf->train(dataset->getSamples(), cv::ml::SampleTypes::ROW_SAMPLE, dataset->getResponses());
    
    clf->save("clf.ext");
    
    std::cout << "Training done!\n";

    return 0;

}
