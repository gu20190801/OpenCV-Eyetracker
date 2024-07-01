#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "ofxGui.h"


using namespace ofxCv;
using namespace cv;

class ofApp : public ofBaseApp{

    public:
        void setup();
        void update();
        void draw();
        void keyPressed(int key);
        void keyReleased(int key);
        void mouseMoved(int x, int y );
        void mouseDragged(int x, int y, int button);
        void mousePressed(int x, int y, int button);
        void mouseReleased(int x, int y, int button);
        void mouseEntered(int x, int y);
        void mouseExited(int x, int y);
        void windowResized(int w, int h);
        void dragEvent(ofDragInfo dragInfo);
        void gotMessage(ofMessage msg);
    
        ofImage img;
        ofImage importImage;
        ofImage eyeLeft;
        ofImage eyeRight;
    
        Mat mat;
        Mat matGrey;
        
        CascadeClassifier face_cascade;
        std::vector<cv::Rect> faces;
    
        CascadeClassifier eye_cascade;
    
        std::vector<cv::Rect> eyes;
    
        ofVideoGrabber myVideoGrabber;
    
        int centerFaceX;
        int centerFaceY;
    
        int centerEyeX;
        int centerEyeY;
        
        int middleOfEyeX;
        int middleOfEyeY;
        int EyeCenterX[2] = {0};
        int EyeCenterY[2] = {0};
        
        int pupilHeight;
    
        int steps;
    
        float ballX = 960;
        float ballY = 540;
    
        ofxCvContourFinder contourFinder;
        
        int LookingCenter[5];
        int LookingTopLeft[5];
        int LookingTopRight[5];
        int LookingBottomLeft[5];
        int LookingBottomRight[5];
        
        ofxPanel gui;
        ofxButton importButton;
        void importButtonPressed();
        
        ofParameter<int> thresholdUpperValue;
        ofParameter<int> thresholdLowerValue;
};
