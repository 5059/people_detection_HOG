/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: generate_dataset.cpp
    Last modifed:   30.12.2016 by Leonardo Citraro
    Description:    Generates a dataset of persons and non-persons in order
                    to train a learning machine

    =========================================================================

    =========================================================================
*/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <math.h>

cv::RNG rng( 3385489 );

// get all the substrings between two chars c1 and c2
std::vector<std::string> get_substr(const std::string str, const char c1, const char c2){
    std::vector<std::string> substrs;
    size_t pos1 = 0;
    size_t pos2 = -1;
    while(1) {
        pos1 = str.find_first_of(c1, pos2+1);
        if(pos1==std::string::npos) 
            break;
        pos2 = str.find_first_of(c2, pos1+1);
        if(pos2==std::string::npos) 
            break;
        substrs.push_back(str.substr( pos1+1, pos2-pos1-1 ));
    };
    return substrs;
}

// parse a comma-separated series of integeres into an array (in this case a 4 points box coordinates)
std::pair<cv::Point,cv::Point> parse_to_array(const std::string str){
    std::vector<int> v;
    std::stringstream ss(str);
    int x;
    while(ss >> x) {
        v.push_back(x);
        if(ss.peek() == ',')
            ss.ignore();
    }
    return std::make_pair<cv::Point,cv::Point>(cv::Point(v[0],v[1]),cv::Point(v[2],v[3]));
}

// in these vectors we store the location of the pedestrians and the image names
// The locations are stored in the annotation files
std::vector<std::vector<std::pair<cv::Point,cv::Point>>> people_coords;
std::vector<std::string> img_names;

// read the annotations files and store the data
void get_annotations(const std::string& str) {
    std::ifstream file(str);
    
    std::string line;
    while(getline(file,line)) {
        std::string name = get_substr(line,'"','"')[0];
        img_names.push_back(name);
        std::vector<std::string> str_coords = get_substr(line,'(',')');
        std::vector<std::pair<cv::Point,cv::Point>> coords;
        for(auto& c:str_coords)
            coords.push_back(parse_to_array(c));
        people_coords.push_back(coords);
    }
    file.close();
}

int main(int argc, char* argv[]) {
    
    // cleanup
    std::system("rm -rdf ./persons/*; mkdir persons");
    std::system("rm -rdf ./not_persons/*; mkdir not_persons");

    // get the training data form the sequence no. 2 only
    // sequence no. 1 will be our test set
    get_annotations("./iccv07-data/annotations/pedxing-seq2-annot.idl.txt");
    
    // size of the box delimiting a person
    cv::Size person_size(40,120);
    
    int count_persons = 0;
    int count_not_persons = 0;
    
    // iterate over all images in the sequence
    for(int i=0; i<img_names.size(); i += 1) {
        std::string name = img_names[i];
        cv::Mat image = cv::imread("./iccv07-data/images/"+name, CV_8U);
        
        // store the location of the persons as cv::Rect
        std::vector<cv::Rect> rec_list;
        
        for(int j=0; j<people_coords[i].size(); ++j) {
            try {
                // person location
                std::pair<cv::Point,cv::Point> c = people_coords[i][j];
                
                // box_size is the size of the box of the annotations files thus different from our person_size.
                // Therefore, we have to adjust the person_size box location
                cv::Size box_size(c.second.x-c.first.x, c.second.y-c.first.y);
                cv::Rect r = cv::Rect(  c.first.x-(person_size.width-box_size.width)/2, 
                                        c.first.y-(person_size.height-box_size.height)/2,
                                        person_size.width,
                                        person_size.height);

                // extract the person sub-image and save
                cv::Mat person = cv::Mat(image, r);
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                rec_list.push_back(r);
                
                // data augmentation
                // here we save other images of the same person but with the box/window slightly shiffted
                // in order to have more training sampels
                /*
                person = cv::Mat(image, cv::Rect(r.x + person_size.width/8,r.y,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                
                person = cv::Mat(image, cv::Rect(r.x + person_size.width/8,r.y + person_size.height/8,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                
                person = cv::Mat(image, cv::Rect(r.x,r.y + person_size.height/8,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                
                person = cv::Mat(image, cv::Rect(r.x - person_size.width/8,r.y,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                
                person = cv::Mat(image, cv::Rect(r.x,r.y - person_size.height/8,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                
                person = cv::Mat(image, cv::Rect(r.x - person_size.width/8,r.y - person_size.height/8,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                
                person = cv::Mat(image, cv::Rect(r.x + person_size.width/8,r.y - person_size.height/8,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                
                person = cv::Mat(image, cv::Rect(r.x - person_size.width/8,r.y + person_size.height/8,r.width,r.height));
                cv::imwrite("./persons/" + std::to_string(count_persons++) + ".jpg", person);
                */
            } catch(...) {
                ;
            }
            
        }
        
        while(1){
            // get a random box
            int x = rng.uniform( 0, image.cols-person_size.width );
            int y = rng.uniform( 0, image.rows-person_size.height );
            cv::Rect r = cv::Rect(cv::Point(x,y), cv::Point(x+person_size.width,y+person_size.height));
            
            // we have to verify that in this random box there isn't a person
            bool intersects;
            for(auto h:rec_list) {
                intersects = false || ((h & r).area() > 0);
            }
            // skip this loop if the random box is over a person box
            if(intersects) continue;
            
            cv::Mat not_person = cv::Mat(image, r);
            cv::imwrite("./not_persons/" + std::to_string(count_not_persons++) + ".jpg", not_person);
            
            // continue saving new not_person images until we reach 120 images
            if(count_not_persons%120 == 0) break;
        }
    }
    
    std::cout << "\n not_persons=" << count_not_persons << " persons=" << count_persons << "\n\n";
    
    return 0;
}
