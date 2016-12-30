/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: main.cpp
    Last modifed:   30.12.2016 by Leonardo Citraro
    Description:    People detection

    =========================================================================

    =========================================================================
*/
#include "opencv_io.hpp"
#include "HOG.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <functional>
#include <math.h>
#include <fstream>

static int MatTYPE = CV_32FC1;
using TYPE = float;

template<class T>
auto load_vector( const std::string& filename ) {
    try {
        std::vector<T> v;
        std::ifstream f(filename, std::ios::binary);
        unsigned int len = 0;
        f.read( (char*)&len, sizeof(len) );
        v.resize(len);
        if( len > 0 ) 
            f.read( (char*)&v[0], len * sizeof(T) );
        f.close();
        return v;
    } catch(...) {
        throw;
    }
}

int main(int argc, char* argv[]) {

    DisplayStream display_orig("Original image",600,600); 
    display_orig.open();
    display_orig.add_trackbar("n_boxes", 1, 50);
    display_orig.set_trackbar_value("n_boxes",2);
    display_orig.add_trackbar("eps", 1, 50);
    display_orig.set_trackbar_value("eps",7);
    
    RecordStream record_video("./people_detection_v1_0.mp4", 8, 640, 480);
    record_video.open();
    
    cv::Size window(40,120);
    
    // load models
    cv::Ptr<cv::ml::SVM> clf = cv::Algorithm::load<cv::ml::SVM>("./training/clf.ext");
    HOG hog = HOG::load("./training/hog.ext");
    auto mean = load_vector<TYPE>("./training/mean.ext");
    auto var = load_vector<TYPE>("./training/var.ext");
    
    cv::Mat image, image2;
    
    // read video sequence image by image
    for(int i=1600; i<2200; i += 2){
        
        std::stringstream filename;
        filename << "./dataset/iccv07-data/images/pedxing-seq1/" << std::setfill('0') << std::setw(8) << std::to_string(i) << ".jpg";
        image = cv::imread(filename.str(), CV_32F);
        cv::cvtColor(image, image2, CV_BGR2GRAY);
        
        hog.process(image2);

        // list of rectangles (positive matches)
        std::vector<cv::Rect> list_rect;
        
        #pragma omp parallel num_threads(8)
        {
            #pragma omp for collapse(2)
            for(size_t x=0; x<image.cols-window.width; x += 5){
                for(size_t y=0; y<image.rows-window.height; y += 5){
                    
                    cv::Rect rec = cv::Rect(x,y, window.width, window.height);
                    auto hist = hog.retrieve(rec);
                    
                    // normalization zero-mean unit-variance
                    for(size_t k = 0; k < hist.size(); ++k) {
                        hist[k] -= mean[k];
                        hist[k] /= var[k];
                    }
                    
                    cv::Mat sample = cv::Mat(1, hist.size(), MatTYPE, hist.data());
                    if(clf->predict(sample) == 1) {
                        #pragma omp critical
                        list_rect.push_back(rec);
                    }
                    
                }
            }
        }
        
        // non-max suppression
        cv::groupRectangles(list_rect, display_orig.get_trackbar_value("n_boxes"), display_orig.get_trackbar_value("eps")/100.f);
        
        // draw rectangles
        for(auto& rec:list_rect) {
            cv::rectangle(image, rec, cv::Scalar(255,255,255), 2);
        }

        display_orig.next_frame(image);
        record_video.next_frame(image);

        if(cv::waitKey(5) == 27){
            break; 
        }
    }
    
    record_video.close();
    
    return 0;
}
