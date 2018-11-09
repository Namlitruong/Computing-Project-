//==============================================================================================================
// Name        : Project.cpp
// Author      : Truong Phan Hoai Nam
// Version     : Final
// Copyright   : Nam Truong
// Description :	This program apply in a surveillance street camera, which have ability to count
//		the amount of vehicles, measuring velocity and classify vehicles by their size, every time having
//		an objects passed through a region that can set manually by users (position, size and color)
//		as well as store the image of that vehicles as a JPEG file. When a vehicles pass by, a region will
//		change into green, therefore, when setting a region make sure it's color is not green or have near
//		green value for better observation. Additionally, if users prefer to observe a signal or display
//		vehicles passed by in a separate window, users can call "Graph" function and "DisplayPassObject"
//		function respectively to have above features. Furthermore, if the program also can store the image of
//		vehicles passed by as well as their data such as the numerical order of that vehicles, which lane is
//		it on, the velocity of that vehicles and the time when that vehicles passed in the HTML file.
//===============================================================================================================
#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <ctime>
#include <stdio.h>
#include <vector>

#include <windows.h>

using namespace cv;
using namespace std;

// Region class
class Region {

	// Private member variables cannot be accessed or viewed from outside the class
private:

	// Mat variables using for background subtraction
	Mat previousFrame;
	Mat currentFrame;
	// Variables for ROI coordinate
	int xROI;
	int yROI;
	int widthROI;
	int heightROI;
	// Variables using for counting function
	bool firstTime;
	int Mean;
	int SumMean;
	int AverageMean;
	int Threshold;
	int ThresholdConst;
	bool below;
	int countObject;
	bool Capture;
	// Variables for storing data in a HTML file
	bool HTMLFirstTime;
	bool writeHTML;
	// Variables for calculating Velocity
	bool Speed;
	float preTime;
	float curTime;
	vector <float> Time;
	// Variables for measuring the length of a vehicle
	bool LengthFlag;
	int MeanLength;
	int SumMeanLength;
	int AverageMeanLength;
	int ThresholdLength;
	int VehiclesLength;
	int preLength;
	int curLength;
	vector <int> Length;


	// Public member variables can be accessed from outside the class, but within the program
	// Those value below of variables can be seem as the initialize value
public:

	Region() {
		xROI = 150;				// Initialize values for counting function
		yROI = 150;
		widthROI = 100;
		heightROI = 100;

		Mean = 0;				// Initialize values for first background subtraction
		SumMean = 0;			// Counting function
		AverageMean = 0;
		Threshold = 0;
		ThresholdConst = 10;
		countObject = 0;

		firstTime = true;		// Using to set up when the function run for the first time
		HTMLFirstTime = true;

		below = true;			// "Flag" for processing just one time when Mean value within
		Capture = false;		// two Threshold values (adaptive threshold and constant threshold),
		writeHTML = false;		// which mean one object passed by and active this functions according to it "Flag"
		Speed = false;
		LengthFlag = false;

		MeanLength = 0;			// Initialize value for second background subtraction
		SumMeanLength = 0;		// Measuring Length function
		AverageMeanLength = 0;
		ThresholdLength = 0;
		VehiclesLength = 0;
	}

	// Function: Setting ROI size in a right position of the lane.

	void setROI (int Input_xROI, int Input_yROI, int Input_widthROI, int Input_heightROI){

		xROI = Input_xROI;
		yROI = Input_yROI;
		widthROI = Input_widthROI;
		heightROI = Input_heightROI;
	}

	/* Function: Drawing ROI in destination frame as well as setting it color and drawing size
	 * 		and if function using to draw a green ROI when object passed
	 * Note: Because when objects passed the ROI will be green, therefore, it will be better
	 * 		when users input color in to this function, it will be better when the color
	 * 		is not green or relate to green.
	*/
	void drawROI (Mat InputFrame, Scalar InputColor, Rect InputSize){

		rectangle (InputFrame, InputSize, InputColor, 5);
		if (Mean > Threshold and Mean > ThresholdConst){
			rectangle (InputFrame, InputSize, Scalar (0, 255, 0), 5);
		}
	}

	// Function: Calculate the mean gray value of the ROI

	void calMean (Mat InputFrame){

	// Declare local variables

		Mat DiffPrevious;
		Mat GrayDiffPrevious;

		int SumWeight = 0;		// reset the value when the program finish calculate value for previous frame
		int Weight;

		if (firstTime){			// Setting frame for the first time that the program run
			InputFrame.copyTo(currentFrame);
			InputFrame.copyTo(previousFrame);
			firstTime = false; // Turn of the first time boolean
		}

		currentFrame.copyTo(previousFrame);		// Setting frame order
		InputFrame.copyTo(currentFrame);

		absdiff(previousFrame, currentFrame, DiffPrevious); // Background subtraction

		cvtColor(DiffPrevious, GrayDiffPrevious, CV_BGR2GRAY); // Convert the Background subtraction into gray

		// First filter: Blur the frame to avoid small noise
		GaussianBlur (GrayDiffPrevious, GrayDiffPrevious, Size (5, 5), 15, 15, BORDER_DEFAULT);

		// Second Filter: Convert gray frame into binary for preventing noise.
		adaptiveThreshold (GrayDiffPrevious, GrayDiffPrevious, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 3 );

		// Read the gray value of each pixels (from 0 to 255) and get the total gray value of the ROI
		for (int y = yROI; y < yROI + heightROI; y++) {
			for (int x = xROI; x < xROI + widthROI; x++){

				Weight = GrayDiffPrevious.at<unsigned char>(y, x);
				SumWeight = SumWeight + Weight;
			}
		}

		Mean = SumWeight / (widthROI * heightROI); // Get the mean gray value of the ROI

	}

	// Function: Calculate threshold based on the mean value

	void calThreshold (int frameCount, int thresholdFactor){

		SumMean = SumMean + Mean;								// Get the total mean value until the current frame
		AverageMean = SumMean / (frameCount + 1);				// Then calculate the average mean value by divide it to the number of current frame
		Threshold = (int)(AverageMean * thresholdFactor); 		// Set the threshold based on AverageMean and ThresholdFactor, which setting by users. For Instance, 1.25 for the threshold will be higher 25% than the average
	}

	// Function: Counting and Capturing image when a vehicle passed

	int Counting (Mat InputFrame){

		if (below == true and Mean > ThresholdConst and Mean > Threshold){	// the below codes will run when a object touch the ROI
			countObject++;							// Counting objects when it arrived
			below = false;							// The "Flag" change the status to make sure just count it once when the Mean > Threshold

		}
		if (below == false and Mean < ThresholdConst and Mean < Threshold){	// Return the "Flag" value when an object gone.
			below = true;
		}
	// Return the number of Objects passed
	return countObject;
	}

	// Function: Put a counter in a ROI and set it color

	void PutTextInROI (Mat InputFrame, Scalar InputColor){

		putText( InputFrame, format("%d", countObject), Point(xROI + (widthROI/4), yROI + ((3*heightROI)/4)), FONT_HERSHEY_PLAIN, 4, InputColor, 7, 5);
	}

	// Function: Showing signal graph in a separate window

	void Graph (Mat InputGraph, string InputGrapWindow,int frameCount){

		line (InputGraph, Point(frameCount % 300, 0), Point(frameCount % 300, 256), Scalar (0, 0, 0), 1);
		line (InputGraph, Point(frameCount % 300, 0), Point (frameCount % 300, Mean), Scalar (0, 255, 0), 1);
		circle (InputGraph, Point (frameCount % 300, Threshold), 1, Scalar(0, 0, 255), 1);
		flip (InputGraph, InputGraph, 0);
		imshow (InputGrapWindow, InputGraph);
		flip (InputGraph, InputGraph, 0);
	}

	/* Function: Display a passing by vehicles in a separate window
	 * 	the "DisplayPassObject" function is quite the same as "CoutingAndCapturing" function
	 * 	just replace imwrite to save the image into inshow to show the image. The reason
	 * 	it in the separate function, so that users can have a choice to choose whether
	 * 	they want to see the image of an passing object.
	 */

	void CapturePassObject (Mat InputFrame, string InputObPassWindow, char InputFileName[100], char InputSaveImage[100]){

		if (Capture == false and Mean > ThresholdConst and Mean > Threshold){
			Capture = true;
			sprintf(InputFileName, InputSaveImage, countObject);	// Saving image according to object numbers
			/*
			 * Crop the image of a vehicle, because the function count a object when it touch the ROI,
			 * so that the rectangle that capture the vehicle need to be above the ROI
			 */
			Mat Capture (InputFrame, Rect (xROI, yROI - heightROI, widthROI + 150, heightROI + 250));

			imwrite (InputFileName, Capture);		// Save an image
			imshow (InputObPassWindow, Capture);
		}
		if (Capture == true and Mean < ThresholdConst and Mean < Threshold){
			Capture = false;

		}
	}

	// Function: Create and store data in a HTML file

	void InputHTML (FILE *InputFilePointer, char InputFileName[100], char InputHTMLImage[100], char LaneName[50], float Speed){

		if (HTMLFirstTime){			// When the program run for the first time, the codes below create a HTML file and the HTML body

			InputFilePointer = fopen("objects.html","w");
			fprintf( InputFilePointer,"<html>\n   <body>\n");
			fclose(InputFilePointer);
			HTMLFirstTime = false;
		}

		if (writeHTML == false and Mean > ThresholdConst and Mean > Threshold){ 	// The codes below will run when an object touch the ROI
			writeHTML = true;							// The "Flag" change the status to make sure just write a line once for an object when the Mean > Threshold
			InputFilePointer = fopen("objects.html","a");	// open file by using "a" (add more line into the file)
			sprintf( InputFileName, InputHTMLImage, countObject); // Because the file name according to it order, therefore using to recall the file name of the stored image

			time_t now = time(0);		// Get the time of the system
			char*date = ctime(&now);	// Convert time into understandable time format

			fprintf( InputFilePointer, "<img src=%s><br>\n" ,InputFileName);						// First line in HTML: Image of a vehicles
			fprintf( InputFilePointer, "Vechicle number %d in %s <br>\n", countObject, LaneName);	// Second line in HTML: Display vehicle numbers and it's lane
			fprintf( InputFilePointer, "%s <br>\n" , date);											// Third Line in HTML: Time that vehicles passed
			fprintf( InputFilePointer, "Speed: %.2f Km/h <br>\n" , Speed);							// Fourth Line in HTML: Showing a Vehicles Velocity


		// Third line in HTML: Show current time
			fclose(InputFilePointer);				// Closing file
		}

		if (writeHTML == true and Mean < ThresholdConst and Mean < Threshold){// Return the "Flag" value when an object gone.
//			InputFilePointer = fopen("objects.html","a");
			writeHTML = false;
		}
	}

	// Function: Capturing Time when a vehicles passed

	vector<float> CapTime(int frameCount, float Timer){

		float subTimer, diffTime;
		// Take out the time when a vehicle touch the ROI
		if (Speed == false and Mean > ThresholdConst and Mean > Threshold){
			subTimer = Timer;
			Speed = true;
		}
		// Return the "Flag" value when a vehicle passed
		if (Speed == true and Mean < ThresholdConst and Mean < Threshold){
			Speed = false;
		}


		preTime = subTimer;			// Setting previous time
		if (preTime != curTime){	// When current time difference to previous time it will take out the time again, because in the above condition
			diffTime = subTimer;	// the time taking out is kind of range of number, through this condition, the program just take out one
		}else{
			diffTime = 0;			// Give others value equal to zero
		}
		curTime = subTimer;			// Setting current time

		if ((int)diffTime != 0){	// When the difference time not equal to zero
			Time.push_back (diffTime);	// this condition will adding value into Time vector,
		}								// By this ways, the program can remember the order of the time corresponding to the vehicle pased by

	return Time;	// returning the vector for Speed function (out side of this class)
	}

	// Function: Showing vehicles speed

	void ShowSpeed (Mat InputFrame, float InputSpeed, Scalar InputColor){
		putText( InputFrame, format ("%.2f Km/h", InputSpeed), Point (xROI , yROI + (heightROI + 25)), FONT_HERSHEY_PLAIN, 2, InputColor, 3, 5);
		if (Mean > Threshold and Mean > ThresholdConst){ // Changing text color into green when a vehicle pased by
			putText( InputFrame, format ("%.2f Km/h", InputSpeed), Point (xROI , yROI + (heightROI + 25)), FONT_HERSHEY_PLAIN, 2, Scalar (0, 255, 0), 3, 5);
		}
	}

	// Function: Using another background subtraction for capturing the length of a vehicle

	int CapLength (Mat InputFrame, int frameCount, float Speed){

		Mat DiffPrevious;
		Mat GrayDiffPrevious;
		Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // declare structure elements for filling the gap with in the white boudary
		// Local variables for second background subtraction and measuring vehicle length
		int LengthLocal = 0;
		int subLengthLocal = 0;
		int SumWeight = 0;		// reset the value when the program finish calculate value for previous frame
		int Weight;

		absdiff(previousFrame, currentFrame, DiffPrevious); // Background subtraction

		cvtColor(DiffPrevious, GrayDiffPrevious, CV_BGR2GRAY); // Convert the Background subtraction into gray

		// First filter: Blur the frame to avoid small noise
		GaussianBlur (GrayDiffPrevious, GrayDiffPrevious, Size (5, 5), 15, 15, BORDER_DEFAULT);

		// Second Filter: Convert gray frame into binary for prevent noise.
		threshold(GrayDiffPrevious, GrayDiffPrevious, 9, 255, THRESH_BINARY);

		// Filling the black area with in an objects for continuity signal
		dilate (GrayDiffPrevious, GrayDiffPrevious, structuringElement);

		// Read the gray value of each pixels (from 0 to 255) and get the total gray value of the ROI
		for (int y = yROI; y < yROI + heightROI; y++) {
			for (int x = xROI; x < xROI + widthROI; x++){
				Weight = GrayDiffPrevious.at<unsigned char>(y, x);
				SumWeight = SumWeight + Weight;
			}
		}

		MeanLength = SumWeight / (widthROI * heightROI); // Get the mean gray value of the ROI
		SumMeanLength = SumMeanLength + MeanLength;
		AverageMeanLength = SumMeanLength / (frameCount + 1);
		ThresholdLength = (int)(AverageMeanLength * 3);	// Setting threshold for this background with the threshold factor is 3

		// Start to count the length of a vehicles when it touch the ROI until it leave
		if (MeanLength > ThresholdLength and MeanLength > 50){
			VehiclesLength ++;
		}
		// Return the length value into zero for next coming vehicle
		if (MeanLength < ThresholdLength and MeanLength < 50 ){
			VehiclesLength = 0;
		}
		// Put the length into a vector to put it separate with the updating variable above
		if (VehiclesLength != 0){
			Length.push_back (VehiclesLength);
		}
		// When the counting function is done, basing on a vehicle speed to shrink or stretch the signal
		if (VehiclesLength == 0){
			if (Speed < 50.0){
				subLengthLocal = Length.size() - 4;
			}
			if (Speed >= 50.0 and Speed < 60.0){
				subLengthLocal = Length.size() - 2;
			}
			if (Speed >= 60.0 and Speed < 70.0){
				subLengthLocal = Length.size();
			}
			if (Speed >= 70.0 and Speed < 80.0){
				subLengthLocal = Length.size() + 2;
			}
			if (Speed >= 80.0 and Speed < 90.0){
				subLengthLocal = Length.size() + 4;
			}
			if (Speed >= 90.0){
				subLengthLocal = Length.size() + 6;
			}
		Length.resize(0, 0);	// Resize the vector when finish compute the length and put it in subLengthLocal
		}
		// Update new length into another variables and setting condition for avoiding collision
		if (subLengthLocal != 0 and subLengthLocal > 6){
			LengthLocal = subLengthLocal;
		}
		return LengthLocal; // Returning the length value for classify condition in main function
	}
};


//Function: Counter, using as a timer for the program base on Video frame rate

float counter (int frameCount, float InputTimer, float FrameRatio){
	if (frameCount % 1 == 0){
		InputTimer = FrameRatio + InputTimer;
	}
	return InputTimer;
}

// Function: Measuring speed for a vehicles
// The reason for putting this out side the main because it use more information of two objects of the class

float Speed (vector <float> InputT1, vector <float> InputT2, float Speed){
	// The actual Distance is 4.8 meter
	// coef value for converting m/s to km/h
	float Time, Distance = 4.8, coef = 3.6;
	// The function will start to calculate the speed of a vehicles when it have t1 and t2 for subtraction
	// Furthermore, the size of the vector have to difference to zero for avoiding out of range error
	if (InputT1.size() == InputT2.size() and InputT1.size() != 0){
		Time = InputT1.at(InputT1.size()-1) - InputT2.at(InputT2.size()-1); // Calculate the time
		Speed = (Distance/Time)*coef;	// Calculate the speed
	}
	return Speed; // Return speed information for Length calculating
}

// Main function

int main(  int argc, char** argv ) {

	// Declare frame Mat that using below
	Mat RawFrame;
	Mat DrawFrame;
	Mat GraphLane1 = Mat(256,300, CV_8UC3);
	Mat GraphLane2 = Mat(256,300, CV_8UC3);

	// Declare variables
	int frameCount;
	float FrameRatio = 1.0/30.0; // Frame rate, this mean that the format of the video is 30 FPS
	float timer = 0;			// Initialize value of the timer
	float Lane1Speed = 0.0;		// Buffer for lane 1 vehicles speed
	float Lane2Speed = 0.0;		// Buffer for lane 2 vehicles speed

	int Lane1Length;			// Buffer for lane 1 vehicles length
	int Lane2Length;			// Buffer for lane 1 vehicles length

	// Declare variables for classifying
	int Big = 0;
	int Small = 0;
	int Medium = 0;
	// Boolean for pause the video
	bool pause;
	// Buffer to store data that use in above function
	char FileLane1[100];
	char FileLane2[100];

	// File pointer for "InputHTML" function
	FILE *file;

	//Create objects of class Region
	Region Lane1;
	Region Lane2;
	Region Lane11;
	Region Lane22;

	// Setting position and size for each objects
	Lane1.setROI (375, 515, 200, 200);
	Lane2.setROI (650, 515, 220, 200);
	Lane11.setROI (425, 360, 140, 100);
	Lane22.setROI (610, 360, 150, 100);
	// Capturing frame from a video for simulate, it can be replace by a surveillance camera
	string VideoFileName = ("ProjectVideo.mp4");
	VideoCapture cap (VideoFileName);
	// Check if the video open successfully
	if (!cap.isOpened()){
		return -1;
	}

	// Creating windows to display "Mat"
	namedWindow("Camera", CV_WINDOW_NORMAL);
	namedWindow("GraphLane1", CV_WINDOW_NORMAL);
	namedWindow("GraphLane2", CV_WINDOW_NORMAL);
	namedWindow("PassObjectLane1", CV_WINDOW_NORMAL);
	namedWindow("PassObjectLane2", CV_WINDOW_NORMAL);

	for (frameCount = 1; frameCount < 1000000000; frameCount++){

		timer = counter (frameCount, timer, FrameRatio);

		// If the video end, open it again

		// Capturing frame from RawFrame Mat
		cap >> RawFrame;

		if (cap.read(RawFrame) == NULL){
			cap.open(VideoFileName);
			cap >> RawFrame;
		}

		/* Copy frame from RawFrame to DrawFrame
		 * DrawFrame use for displaying ROI, number of vehicles, current time.
		 * RawFrame use for background subtraction and capturing image
		 */
		RawFrame.copyTo(DrawFrame);

		// Call functions for object Lane1
		Lane1.drawROI( DrawFrame, Scalar(0,0,255), Rect (375, 515, 200, 200));							// Drawing a red ROI in a DrawFrame
		Lane1.calMean(RawFrame);																		// Calculate Mean value in a RawFrame
		Lane1.calThreshold(frameCount, 5);																// Setting Threshold value
		Lane1.Counting(RawFrame);																		// Counting vehicles passed
		Lane1.PutTextInROI(DrawFrame, Scalar (0, 0, 255));												// Put counter inside the ROI for observation
		Lane1.Graph (GraphLane1, "GraphLane1",frameCount);												// Showing Graph signal
		Lane1.CapturePassObject(RawFrame, "PassObjectLane1", FileLane1, "Lane1-%d.JPEG");				// Capturing and showing image vehicle when it passed

		Lane11.drawROI( DrawFrame, Scalar(0,0,255), Rect (425, 360, 140, 5));							// Drawing ROI for Lane11
		Lane11.calMean(RawFrame);																		// Calculate Mean value in a RawFrame
		Lane11.calThreshold(frameCount, 5);																// Setting Threshold value
		Lane11.Counting(RawFrame);																		// Counting vehicles passed

		Lane1Speed = Speed (Lane1.CapTime(frameCount, timer), Lane11.CapTime(frameCount, timer), Lane1Speed);	// Measuring Speed of lane 1 vehicles
		Lane1Length = Lane1.CapLength(RawFrame, frameCount, Lane1Speed);										// Calculating length of lane 1 vehicles
		Lane1.ShowSpeed (DrawFrame, Lane1Speed, Scalar (0, 0, 255));											// Showing Vehicles speed
		Lane1.InputHTML(file, FileLane1, "Lane1-%d.JPEG", "Lane 1", Lane1Speed);								// Saving data in a HTML file



		// Call functions for object Lane2
		Lane2.drawROI( DrawFrame, Scalar(255,0,0), Rect (650, 515, 220, 200));							// Drawing a blue ROI in a DrawFrame
		Lane2.calMean(RawFrame);																		// Calculate Mean value in a RawFrame
		Lane2.calThreshold(frameCount, 5);																// Setting Threshold value
		Lane2.Counting(RawFrame);																		// Counting vehicles passed
		Lane2.PutTextInROI(DrawFrame, Scalar (255, 0, 0));												// Put counter inside the ROI for observation
		Lane2.Graph (GraphLane2, "GraphLane2",frameCount);												// Showing Graph signal
		Lane2.CapturePassObject(RawFrame, "PassObjectLane2", FileLane2, "Lane2-%d.JPEG");				// Showing image vehicle when it passed

		Lane22.drawROI( DrawFrame, Scalar(255,0,0), Rect (610, 360, 150, 5));							// Drawing ROI for lane 22
		Lane22.calMean(RawFrame);																		// Calculate Mean value in a RawFrame
		Lane22.calThreshold(frameCount, 5);																// Setting Threshold value
		Lane22.Counting(RawFrame);																		// Counting vehicles passed

		Lane2Speed = Speed (Lane2.CapTime(frameCount, timer), Lane22.CapTime(frameCount, timer), Lane2Speed);	// Measuring Speed of Lane 2 Vehicles
		Lane2Length = Lane2.CapLength(RawFrame, frameCount, Lane2Speed);										// Calculating Length for Lane 2 Vehicles
		Lane2.ShowSpeed (DrawFrame, Lane2Speed, Scalar (255, 0, 0));											// Showing Speed for Lane 2 Vehicles
		Lane2.InputHTML(file, FileLane2, "Lane2-%d.JPEG", "Lane 2", Lane2Speed);								// Saving data in a HTML file

		// Classifying Vehicles by their size in Lane 1
		if (Lane1Length != 0){
			if (Lane1Length >= 21){ // If the signal greater than 21, it is a big size vehicle
				Big ++;
				putText( DrawFrame, "Big", Point (375, 515), FONT_HERSHEY_PLAIN, 5, Scalar (0, 0, 255), 10, 5);
			}
			if (Lane1Length >= 14 and Lane1Length < 21){ // If the signal from 14 to 21, it is a medium size vehicle
				Medium ++;
				putText( DrawFrame, "Medium", Point (375, 515), FONT_HERSHEY_PLAIN, 5, Scalar (0, 0, 255), 10, 5);
			}
			if (Lane1Length < 14){ // If the signal less than 14, it is a small size vehicle
				Small ++;
				putText( DrawFrame, "Small", Point (375, 515), FONT_HERSHEY_PLAIN, 5, Scalar (0, 0, 255), 10, 5);
			}
		}

		// Classifying Vehicles by their size in Lane 2
		if (Lane2Length != 0){
			if (Lane2Length >= 21){ // If the signal greater than 21, it is a big size vehicle
				Big ++;
				putText( DrawFrame, "Big", Point (650, 515), FONT_HERSHEY_PLAIN, 5, Scalar (255, 0, 0), 10, 5);
			}

			if (Lane2Length >= 14 and Lane2Length < 21){// If the signal from 14 to 21, it is a medium size vehicle
				Medium ++;
				putText( DrawFrame, "Medium", Point (650, 515), FONT_HERSHEY_PLAIN, 5, Scalar (255, 0, 0), 10, 5);
			}

			if (Lane2Length < 14){// If the signal less than 14, it is a small size vehicle
				Small ++;
				putText( DrawFrame, "Medium", Point (650, 515), FONT_HERSHEY_PLAIN, 5, Scalar (255, 0, 0), 10, 5);
			}

		}
		// Drawing Black box for better observation of Text in it
		rectangle (DrawFrame, Rect (0,0,550,100), Scalar (0, 0, 0), -5);

		// Print the information into DrawFrame
		putText( DrawFrame, format("Big = %d, Medium = %d, Small = %d", Big, Medium, Small), Point(5, 75), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 5, 5);

		time_t now = time(0);		// Get the time of the system
		char*date = ctime(&now);	// Convert time into understandable time format
		putText( DrawFrame, format("%s", date), Point(5,25), FONT_HERSHEY_PLAIN, 2, Scalar(250,255,255), 5, 5); // Display time into DrawFrame

		imshow("Camera", DrawFrame);	// Showing Mat DrawFrame in a "Camera" window

		switch(waitKey(10)){

		case 27: // Pressing "Esc" key to exit the program
			return 0;

		case 112: //'p' has been pressed. this will pause/resume the code.
			pause = !pause;
			if(pause == true){
				printf ("Code Paused \n");
			while (pause == true){
				//Stay in this loop until
				switch (waitKey()){
				case 112:
					//Change pause back to false.
					pause = false;
					printf ("Code resumed \n");
					break;
				}
			}
			}
		}
	}

	return 0;
}
