#include "ofApp.h"
#include <algorithm>

void ofApp::setup(){
    // GUI Setup
    gui.setup();
    gui.setPosition(700, 250);
    gui.add(thresholdUpperValue.set("Pupil Detection Upper Threshold", 255, 128, 255));
    gui.add(thresholdLowerValue.set("Pupil Detection Lower Threshold", 50, 0, 128));
    importButton.addListener(this, &ofApp::importButtonPressed);
    gui.add(importButton.setup("Import Image"));
    
    // Get webcam input
    cout<<myVideoGrabber.listDevices().data()<<endl;
    myVideoGrabber.setDeviceID(1);
    myVideoGrabber.initGrabber(640, 480);
    
    //need to use absolute path here, for the latest version of OpenFrameworks and its OpenCV addon
    if(!face_cascade.load("/Users/user/Desktop/of_v0.12.0_osx_release/apps/myApps/miniProject/bin/data/haarcascade_frontalface_default.xml")){
        cout<<"Error loading"<<endl;
    }
    if(!eye_cascade.load("/Users/user/Desktop/of_v0.12.0_osx_release/apps/myApps/miniProject/bin/data/haarcascade_eye_tree_eyeglasses.xml")){
        cout<<"Error loading"<<endl;
    }
    
    centerFaceX = 0;
    centerFaceY = 0;
}

//--------------------------------------------------------------
void ofApp::update(){
    myVideoGrabber.update();
    if (myVideoGrabber.isFrameNew()) {
        // Obtain a pointer to the grabber's image data
        img.setFromPixels(myVideoGrabber.getPixels());
        mat = toCv(img);
        cvtColor(mat, matGrey, CV_BGR2GRAY);
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofSetColor(255, 255, 255, 255);
    ofNoFill();
    drawMat(mat, 0, 0);
    
    // Face & Eye detection
    face_cascade.detectMultiScale(mat, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    eye_cascade.detectMultiScale(mat, eyes, 1.1, 2, 0|CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    centerFaceX = 0;
    centerFaceY = 0;
    int maxFaceROI = 0;

    // Draw Face Blob
    for(int i = 0; i < faces.size(); i++){
        ofNoFill();
        int faceWidth = faces.at(i).width;
        int faceHeight = faces.at(i).height;
        int faceROI = faceWidth * faceHeight;
        if (faceROI >= maxFaceROI){
            centerFaceX = faces.at(i).x + faces.at(i).width / 2;
            centerFaceY = faces.at(i).y + faces.at(i).height / 2;
        }
        ofDrawRectangle(faces.at(i).x, faces.at(i).y, faces.at(i).width, faces.at(i).height);
    }
    
    // Eye Tracking
    if (eyes.size() == 2){
        for (int i = 0; i < eyes.size(); i++) {
            // Calculate the center point of the eye region
            int centerX = eyes.at(i).x + eyes.at(i).width / 2;
            int centerY = eyes.at(i).y + eyes.at(i).height / 2;
            EyeCenterX[i] = centerX;
            EyeCenterY[i] = centerY;
            if (centerY <= centerFaceY){
                ofSetColor(255, 0, 0, 255);
                ofDrawCircle(centerX, centerY, 5);
                
                // Base on the eye region we get, crop eye regions
                ofImage tempImg = img;
                tempImg.crop(eyes[i].x, eyes[i].y+eyes[i].height*.3, eyes[i].width, eyes[i].height*.4);
                ofSetColor(255, 255, 255, 255);
                
                if (eyes[i].x < centerFaceX){
                    // Left Eye
                    eyeLeft = tempImg;
                }
                else{
                    // Right Eye
                    eyeRight = tempImg;
                }
                
                // Only use Left Eye for further computer vision processing
                eyeLeft.resize(500, 200);
                Mat eyeLeftMat;
                eyeLeftMat = toCv(eyeLeft);
                if (!eyeLeftMat.empty()){
                    Mat GrayMat;
                    cvtColor(eyeLeftMat, GrayMat, CV_BGR2GRAY);
                    threshold(GrayMat, GrayMat, thresholdLowerValue, thresholdUpperValue, THRESH_BINARY_INV);
                    
                    // Perform erosion and dilation
                    int erosionSize = 3;
                    Mat element = getStructuringElement(MORPH_ELLIPSE, cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1), cv::Point(erosionSize, erosionSize));

                    erode(GrayMat, GrayMat, element);
                    dilate(GrayMat, GrayMat, element);

                    // Find contours in the image
                    vector<vector<cv::Point>> contours;
                    vector<Vec4i> hierarchy;
                    findContours(GrayMat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                    for (size_t i = 0; i < contours.size(); i++) {
                        double area = contourArea(contours[i]);
                        if (area > 1000) {
                            Moments contourMoments = moments(contours[i]);
                            if (contourMoments.m00 != 0) {
                                // Calculate the center point of the contour
                                cv::Point center(static_cast<int>(contourMoments.m10 / contourMoments.m00), static_cast<int>(contourMoments.m01 / contourMoments.m00));
                                // Draw contour blob
                                cv::Rect boundingRect = cv::boundingRect(contours[i]);
                                drawContours(eyeLeftMat, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
                                cv::rectangle(eyeLeftMat, boundingRect, cv::Scalar(0, 0, 255), 2);
                                pupilHeight = boundingRect.height;
                                circle(eyeLeftMat, center, 5, Scalar(0, 0, 255), FILLED);
                                centerEyeX = static_cast<int>(center.x);
                                centerEyeY = static_cast<int>(center.y);
                                cout << center << endl;
                            }
                        }
                    }
                    drawMat(eyeLeftMat, 640, 0);
                }
            }
            
            // Get the coordinates between two eyes
            middleOfEyeX = (EyeCenterX[0]+EyeCenterX[1])/2;
            middleOfEyeY = (EyeCenterY[0]+EyeCenterY[1])/2;
            ofSetColor(255, 255, 0, 255);
            ofDrawLine(middleOfEyeX-15, middleOfEyeY, middleOfEyeX+15, middleOfEyeY);
            ofDrawLine(middleOfEyeX, middleOfEyeY-15, middleOfEyeX, middleOfEyeY+15);
        }
    }
    
    // Text output
    ofSetColor(255, 255, 255, 255);
    ofFill();
    ofDrawBitmapStringHighlight("Looking at Center: " + ofToString(LookingCenter[0]) + ", " + ofToString(LookingCenter[1]) + ", " + ofToString(LookingCenter[4]) + "; " + ofToString(LookingCenter[2]) + ", " + ofToString(LookingCenter[3]), 10, 20);
    ofDrawBitmapStringHighlight("Looking at Top-Left: " + ofToString(LookingTopLeft[0]) + ", " + ofToString(LookingTopLeft[1]) + ", " + ofToString(LookingTopLeft[4]) + "; " + ofToString(LookingTopLeft[2]) + ", " + ofToString(LookingTopLeft[3]), 10, 40);
    ofDrawBitmapStringHighlight("Looking at Top-Right: " + ofToString(LookingTopRight[0]) + ", " + ofToString(LookingTopRight[1]) + ", " + ofToString(LookingTopRight[4]) + "; " + ofToString(LookingTopRight[2]) + ", " + ofToString(LookingTopRight[3]), 10, 60);
    ofDrawBitmapStringHighlight("Looking at Bottom-Left: " + ofToString(LookingBottomLeft[0]) + ", " + ofToString(LookingBottomLeft[1]) + ", " + ofToString(LookingBottomLeft[4]) + "; " + ofToString(LookingBottomLeft[2]) + ", " + ofToString(LookingBottomLeft[3]), 10, 80);
    ofDrawBitmapStringHighlight("Looking at Bottom-Right: " + ofToString(LookingBottomRight[0]) + ", " + ofToString(LookingBottomRight[1]) + ", " + ofToString(LookingBottomRight[4]) + "; " + ofToString(LookingBottomRight[2]) + ", " + ofToString(LookingBottomRight[3]), 10, 100);
    ofDrawBitmapStringHighlight("Press 'r' to reset the eye tracking callibration", 1520, 1000);
    
    // Draw indicators for 5-point Eye Tracking Calibration
    if (steps == 0){
        ofSetColor(255, 255, 255, 255);
        ofFill();
        ofDrawCircle(960, 540, 30);
    }
    else if (steps == 1){
        ofSetColor(255, 255, 255, 255);
        ofFill();
        ofDrawCircle(30, 30, 30); // Upper-Left Corner
    }
    else if (steps == 2){
        ofSetColor(255, 255, 255, 255);
        ofFill();
        ofDrawCircle(1890, 30, 30); // Upper-Right Corner
    }
    else if (steps == 3){
        ofSetColor(255, 255, 255, 255);
        ofFill();
        ofDrawCircle(30, 1000, 30); // Lower-Left Corner
    }
    else if (steps == 4){
        ofSetColor(255, 255, 255, 255);
        ofFill();
        ofDrawCircle(1890, 1000, 30); // Lower-Right Corner
    }
    else if (steps >= 5){
        // Draw imported image
        int imgWidth = 1920;
        int imgHeight = 1080;
        if (importImage.isAllocated()){
            imgWidth = importImage.getWidth();
            imgHeight = importImage.getHeight();
            if (imgWidth > 1920 || imgHeight > 1080){
                if (imgHeight >= imgWidth){
                    importImage.resize(imgWidth/(imgHeight/1080), 1080);
                }
                else {
                    importImage.resize(1920, imgHeight/(imgWidth/1920));
                }
            }
            importImage.draw(960-importImage.getWidth()/2, 540-importImage.getHeight()/2);
        }
        
        // Easing Function
        ofSetColor(255, 255, 255, 255);
        ofFill();
        
        // Map function based on pupil blob tracking
        float targetX1 = ofMap(centerEyeX, MIN(LookingTopLeft[0], LookingBottomLeft[0]), MAX(LookingTopRight[0], LookingBottomRight[0]), 0, 1920);
        
        // Map function based on the mid-point between eyes
        float targetX2 = ofMap(middleOfEyeX, MIN(LookingTopLeft[2], LookingBottomLeft[2]), MAX(LookingTopRight[2], LookingBottomRight[2]), 0, 1920);
        float targetY = ofMap(middleOfEyeY, MIN(LookingTopLeft[3], LookingTopRight[3]), MAX(LookingBottomLeft[3], LookingBottomRight[3]), 0, 1080);
        
        
        float resultX = targetX1*.5+targetX2*.5; // Combine two calculated x-coordinates
        float resultY = targetY;
        float easing = 0.05;

        // Update the ball position using easing method
        float dx = resultX - ballX;
        float dy = resultY - ballY;
        ballX += dx * easing;
        ballY += dy * easing;

        // Clamp the ball position to the screen boundaries
        ballX = ofClamp(ballX, 0, 1920);
        ballY = ofClamp(ballY, 0, 1080);

        ofDrawBitmapStringHighlight("Now Looking at: " + ofToString(ballX) + ", " + ofToString(ballY), 10, 120);
        ofSetColor(0, 255, 0, 100);
        ofDrawCircle(ballX, ballY, 100);
    }
    gui.draw();
}

void ofApp::importButtonPressed() {
    // Open a file dialog to allow the user to select an image file
    ofFileDialogResult result = ofSystemLoadDialog("Select an image file");
    if (result.bSuccess) {
        // Get the path of the selected image file
        string imagePath = result.filePath;
        vector<string> imageExtensions = {".jpg", ".jpeg", ".png"};
        string extension = ofToLower(imagePath);
        // File format validation
        for (const string& imageExt : imageExtensions) {
            if (ofIsStringInString(extension, imageExt)) {
                importImage.load(imagePath);
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    // Reset the steps variable and the recorded eye tracking calibration to 0
    if (key == 'r') {
        steps = 0;
        for (int i=0; i<5; i++){
            LookingCenter[i] = 0;
            LookingTopLeft[i] = 0;
            LookingTopRight[i] = 0;
            LookingBottomLeft[i] = 0;
            LookingBottomRight[i] = 0;
        }
    }
}

//------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    // When mouse clicked, Save eye tracking calibration to corresponding arrays
    if (steps == 0){
        LookingCenter[0] = centerEyeX;
        LookingCenter[1] = centerEyeY;
        LookingCenter[2] = middleOfEyeX;
        LookingCenter[3] = middleOfEyeY;
        LookingCenter[4] =  pupilHeight;
    }
    else if (steps == 1){
        LookingTopLeft[0] = centerEyeX;
        LookingTopLeft[1] = centerEyeY;
        LookingTopLeft[2] = middleOfEyeX;
        LookingTopLeft[3] = middleOfEyeY;
        LookingTopLeft[4] =  pupilHeight;
    }
    else if (steps == 2){
        LookingTopRight[0] = centerEyeX;
        LookingTopRight[1] = centerEyeY;
        LookingTopRight[2] = middleOfEyeX;
        LookingTopRight[3] = middleOfEyeY;
        LookingTopRight[4] =  pupilHeight;
    }
    else if (steps == 3){
        LookingBottomLeft[0] = centerEyeX;
        LookingBottomLeft[1] = centerEyeY;
        LookingBottomLeft[2] = middleOfEyeX;
        LookingBottomLeft[3] = middleOfEyeY;
        LookingBottomLeft[4] =  pupilHeight;
    }
    else if (steps == 4){
        LookingBottomRight[0] = centerEyeX;
        LookingBottomRight[1] = centerEyeY;
        LookingBottomRight[2] = middleOfEyeX;
        LookingBottomRight[3] = middleOfEyeY;
        LookingBottomRight[4] =  pupilHeight;
    }
    steps++;
}
//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
