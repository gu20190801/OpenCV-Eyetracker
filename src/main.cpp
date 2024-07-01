#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main( ){
	//Use ofGLFWWindowSettings for more options like multi-monitor fullscreen
	ofGLWindowSettings settings;
	settings.setSize(1920, 1080);
    settings.title = "Tracking Input";
    settings.windowMode = OF_WINDOW; 
	auto mainWindow = ofCreateWindow(settings);
    
    auto mainApp = make_shared<ofApp>();
    
    ofRunApp(mainWindow, mainApp);
	ofRunMainLoop();
}
