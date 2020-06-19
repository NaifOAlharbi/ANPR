#include <iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<sstream>
#include<string>
#include<numeric>
#include<ctime>
#include<cmath>
#include<chrono>
using namespace std;
using namespace cv;




/////////////////////// Start the ANPR system fucntion prototype///////////////////////
void Start_the_ANPR_System(String test_dir, String videofilename);

/////////////////////// Number Plate Detection function prototype ///////////////////////
vector<Mat> Number_Plate_Detection(Mat image, vector<Rect> & Draw_Plates, vector<double> &Colors);
//////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////Number Plate Segmentation and Characters Detection function prototype///////////////////////
void PlateSegmentation_CharatersDetection(Mat, vector<Rect>&, vector<Mat>&);
bool verify_character_size(Mat r);                                                                  // Verify if the detected contour is a character or not
bool Hall(const Rect & a, const Rect & b);                                                           // Organize the detected characters accroding to their location
//////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////// Character Recognition function prototype///////////////////////
bool Characters_Recognition(vector<Rect>, vector<Mat>, String &);
Mat ProcsessChar(Mat in);                                                                           // Process the character image to be same as the training samples









int main(int argc, char **argv) {





	const char* keys =
	{
		"{td    |        | path of directory contains test images}"
		"{tv    |        | test video file name}"


	};

	CommandLineParser parser(argc, argv, keys);
	String test_dir = parser.get< String >("td");
	String videofilename = parser.get< String >("tv");


	if (test_dir.empty() && videofilename.empty())
	{
		parser.printMessage();
		cout << "Wrong number of parameters.\n\n"
			<< "Example command line:\n" << argv[0] << " for pictures please type the file path (-td=path/images/*.jpg) or for video  (-tv=Path/video.mov) or for live stream from the camera (-tv=0) " << endl;

		exit(1);
	}



	Start_the_ANPR_System(test_dir, videofilename);


	return (0);

}


void Start_the_ANPR_System(String test_dir, String videofilename) {
	auto start = chrono::high_resolution_clock::now();
	vector< String > files;             // Get the imgs under the test folder;
	glob(test_dir, files);
	VideoCapture cap;
	int wait;


	if (videofilename != "")
	{

		if (videofilename.size() == 1 && isdigit(videofilename[0]))
			cap.open(videofilename[0] - '0');
		else
			cap.open(videofilename);


	}
	VideoWriter writer;
	if (cap.isOpened()) {
		string name;
		cout << "Please type the name of the video to be saved, without the extension of the video" << endl;
		cin >> name;
		int fcc = CV_FOURCC('m', 'p', '4', 'v');                      //Save the video as mp4

		double fps = cap.get(CV_CAP_PROP_FPS);              // Number of frames for the resulted video
		int framewdith = cap.get(CV_CAP_PROP_FRAME_WIDTH);      // Width of the video
		int frameHight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);    // Hieght of the video
		Size frameSize(framewdith, frameHight);                // adjust the frames width and height
		writer = VideoWriter(name + ".mov", fcc, 25, frameSize);      // Save the video according to user input and the details above

	}

	
	Rect2d N;

	for (size_t i = 0;; i++) //loop in img
	{

	

	
		 //-----------------------Initialize Mat img (color space)---------------------//
	
		Mat img;  //read in image in color space.
		Mat copy;
		Mat CopyOriginal;


		if (cap.isOpened())
		{
			int scaling;                                                       // Declare Scale ratio
			cap >> img;                                                      // read from the capture
			img.copyTo(CopyOriginal);                                        // Copy the original image


			//////////////// This part of the code will generate the ROI in case of live-feed or recoreded videos ////////////////
			if (i == 0) {
				if (img.size().width >= 1920 && img.size().height >= 1080) {                    // if the video has high resolution, which will soo big to be displyed
					Mat Resized;                                                               // Declare matrix to resize the frames of the video
					cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);                                    // adjust the width of the frames to be 640
					cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360);                                   // adjust the height of the frames to be 360
					resize(img, Resized, Size(640, 360), 0, 0, INTER_CUBIC);                  // Resize the frames to 640*360
					N = selectROI(Resized);                                                     // Adjust the ROI according to the user input
					scaling = img.size().width / Resized.size().width;                            // Adjust the scale ratio to be same as the orignial input
					Rect r(scaling*N.x, scaling*N.y, scaling*N.width, scaling*N.height);    // Extract the ROI to the original image
					N = r;                                                                  // Copy the ROI coordinates to Rect R;
				}
				else { N = selectROI(img); }                                                  // else if the video has normal resolution, which fits the on the screen
			}


			if (img.rows != 0 && img.cols != 0) {
				CopyOriginal = img(N);
			}

			wait = 1;
			namedWindow("video", CV_WINDOW_FREERATIO);
		}
		else if (i < files.size())
		{
			img = imread(files[i]);
			img.copyTo(CopyOriginal);
			wait = 0;
			namedWindow("detection", CV_WINDOW_FREERATIO);
		}
		if (img.empty())
		{
			return;
		}
		
		String Characters;              // Declare String of the recognised Characters
		vector<Mat> plates;             // Declare vector of Matrixs for the Number Plate
		vector<Mat>Chars_detected;      // Declare vector of Matrixs for the deteced Characters
		vector<Rect> Draw_Plates;        // Declare vector of Rectangles to Draw on the original image
		vector<Rect>Detected_Chars_coordinations;      // Declare vector of Rectangles for the characters locations
		vector<double> Colors;                          // Declare vector of Double for the colour of the rectangles

		plates = Number_Plate_Detection(CopyOriginal, Draw_Plates, Colors);                     // Detect the number plate



		if (plates.size() != 0) {
			
			for (int j = 0; j < plates.size(); j++) {
				Scalar color = Scalar(0, Colors[j] * Colors[j] * 200, 0);


				PlateSegmentation_CharatersDetection(plates[j], Detected_Chars_coordinations, Chars_detected);
				rectangle(CopyOriginal, Draw_Plates[j], color, CopyOriginal.cols / 400 + 1);//picture
				if (Characters_Recognition(Detected_Chars_coordinations, Chars_detected, Characters)) {
					putText(CopyOriginal, Characters, Point(Draw_Plates[j].x + 33, Draw_Plates[j].y - 20), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);

					
					
					Characters.clear();
				}


				Chars_detected.clear();

			}
			

		}
		
		if (cap.isOpened()) {
			Mat videos;
			img.copyTo(videos);
			cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
			cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
			resize(img, img, Size(640, 360), 0, 0, INTER_CUBIC);
			imshow("video", img);
			writer.write(videos);
		}

		else {
			
			resize(CopyOriginal, img, Size(800, 600));
			imshow("detection", img);
		}
		if (waitKey(wait) == 27)
			break;


	}
	cout << "done" << endl;

	writer.release();
}








//////////////////////   Number Plate Detection   //////////////////////
vector<Mat> Number_Plate_Detection(Mat image, vector<Rect> & g, vector<double> &Colors) {

	HOGDescriptor hog;             // Declare the HOG descriptor
	Mat copy;
	image.copyTo(copy);
	hog.load("Detect_Number_Plate.xml");          // Read from the trained file XML
	vector<Rect>detections;                       // Declare vector of Rectangles for the deteced objects from the HOG-SVM
	vector< double > foundWeights;                // Declare vector of double for found weights of the detected objects
	hog.detectMultiScale(image, detections, foundWeights);  //APPLY HOG-SVM to the input image
	vector<Mat> crop;
	vector<double>h;
	double H = 0;
	vector<Rect> ROI;                                 // Region Of Interest
	vector<Rect> first;
	for (int i = 0; i < detections.size(); i++) {

		if (foundWeights[i] > 0.7) {                   // choose the detections which have values higher than 0.7

			ROI.push_back(detections[i]);
			Colors.push_back(foundWeights[i]);
			

		}


	}

	// If the number of the detected number plate is 1 //
	if (ROI.size() == 1) {
		Mat HSV_Space;                               // Declare matrix for HSV colour space
		Mat Threshold_HSV;                           // Declare matrix for threshold
		HSV_Space = copy(ROI[0]);                    // Extract the number plate (ROI)
		cvtColor(HSV_Space, HSV_Space, CV_BGR2HSV);  // convert it to HSV colour space
		inRange(HSV_Space, Scalar(20, 100, 100), Scalar(30, 255, 255), Threshold_HSV);           // apply threshold to find the yellow colour
		double yellow_precent = ((double)countNonZero(Threshold_HSV) / (HSV_Space.cols*HSV_Space.rows));      // calculate the existence of the yellow colour

		if (yellow_precent > 0) {

			crop.push_back(image(ROI[0]));
			g.push_back(ROI[0]);

		}



	}


	///      Calculate the existence of the yellow colour in the detections , if the number of the detection is more than 1 ///
	else if (ROI.size() > 1) {
		for (int i = 0; i < ROI.size(); i++) {
			Mat HSV;
			Mat thr_HSV;
			Mat HH;
			HSV = copy(ROI[i]);

			HSV.copyTo(HH);
			cvtColor(HSV, HSV, CV_BGR2HSV);
			inRange(HSV, Scalar(20, 100, 100), Scalar(30, 255, 255), thr_HSV);
			double yellow_precent = ((double)countNonZero(thr_HSV) / (HSV.cols*HSV.rows));
			if (yellow_precent > 0.1) {
				first.push_back(ROI[i]);
				
			}
		}
	}
	for (int i = 0; i < first.size(); i++) {
		crop.push_back(image(first[i]));
		g.push_back(first[i]);

	}
	return crop;

}






//////////////////////   Plate Segmentation & Character Detection   //////////////////////
void PlateSegmentation_CharatersDetection(Mat Plate, vector<Rect>&Detected_Chars_coordinations, vector<Mat>&Chars_detected)
{


	Mat Segment;                             // Declare Matrix for Segementing the number plate
	cvtColor(Plate, Segment, CV_BGR2GRAY);      // Convert the extraced Number Plate to Grayscale
	medianBlur(Segment, Segment, 1);               // Apply the Median filter to remove the noise and sharp the edges

	adaptiveThreshold(Segment, Segment, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 10);  // Apply the Adaptive Threshold


	Mat skel(Segment.size(), CV_8UC1, Scalar(0));                                        // Make Matrix with one channel and adjust  the pixels to zero
	Mat temp(Segment.size(), CV_8UC1);                                                  //Make another matrix with one channel
	Mat element = getStructuringElement(MORPH_CROSS, Size(20, 20));                       // Make matrix for Morphological Operations
	Mat CopySegement;                                                                   // Copy the Number plate
	Segment.copyTo(CopySegement);
	bool done;
	do {
		morphologyEx(CopySegement, temp, MORPH_OPEN, element);                            //Apply the morphology on the image
		// imshow("temp",temp);
		bitwise_not(temp, temp);                                                        //Apply the bitwise for the skeleton
		//imshow("not",temp);
		bitwise_and(CopySegement, temp, temp);                                            //Apply And operator for the skeleton
		//imshow("and",temp);
		bitwise_or(skel, temp, skel);                                                     //Apply OR operator for the skeleton
		//imshow("or",temp);
		erode(CopySegement, CopySegement, element);                                       // Apply Erosion Morphology
		//imshow("erode",temp);
		double max;
		minMaxLoc(CopySegement, 0, &max);
		done = (max == 0);
	} while (!done);

	vector<vector<Point>>contours;                                                 //Declare vector of contours

	findContours(skel, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);  // Find the contours from the skeleton result
	//CV_RETR_EXTERNAL

	Mat result;                                                     // Declare Matrix for the result
	Segment.copyTo(result);                                         // Copy the number plate to result matrix
	cvtColor(result, result, CV_GRAY2BGR);                            // Convert the Number Plate to Grayscale

	drawContours(result, contours, -1, Scalar(255, 0, 0), 1);             // Draw the contours on the image


	vector<Point> Point_Contour;                                    // Declare vector points of contour

	vector<Rect>BoundingRect_Contour(contours.size());             // Declare vector of Bounding Rect for  contour

	for (int i = 0; i < contours.size(); i++) {                        // for each contour
		Point_Contour = contours[i];                                       // Save the contours location
		BoundingRect_Contour[i] = boundingRect(Point_Contour);              // Make Rectangle according the contour

	}


	vector<Rect>Chars;                                               // Declare vector of recangles for the detect Chararacter
	vector<int>heightofChars;                                         //  Declare vector of integers for the Characters height
			////////Verify the detect chars//////
	for (int i = 0; i < BoundingRect_Contour.size(); i++) {
		if (verify_character_size(Segment(BoundingRect_Contour[i]))) {            // if the detect contours is a char save the coordination to vector of chars

			Chars.push_back(BoundingRect_Contour[i]);
			heightofChars.push_back(Segment(BoundingRect_Contour[i]).rows);

		}
	}
	sort(Chars.begin(), Chars.end(), Hall);
	////  check the  detected chararcters if there  is a big difference in the height of  a certain character and the others that character will be cancelled by finding the mean value of the all the chararacters's height and if the height of any character is 3 times above or below this value it will be removed/////
	if (Chars.size() != 0) {
		int H;
		int sum_of_elements(0);
		// sum the heights of the detected contours
		sum_of_elements = accumulate(heightofChars.rbegin(), heightofChars.rend(), 0);
		// find the mean value
		H = sum_of_elements / Chars.size();

		int Max;
		int Min;
		Max = H + 3;// Maximum of the difference
		Min = H - 3;//Minimum of the difference
	   // cout<<Chars.size()<<"gggg"<<endl;
		for (int i = 0; i < Chars.size(); i++) {
			// if there is big difference beteween the mean value and the height of
			// the detected contour, remove that contour
			if (Chars[i].height > Max || Chars[i].height < Min) {
				Chars.erase(Chars.begin() + i);
			}
		}
	}

	//If the deteced characters more than 7//
	if (Chars.size() != 0 && Chars.size() > 7) {
		int f1, f2, f3, f4, f5;

		// Measure the distance between the first character and the second//
		f1 = abs(Chars[0].x + Chars[0].width - Chars[1].x);
		f2 = abs(Chars[1].x + Chars[1].width - Chars[2].x);
		f3 = abs(Chars[6].x + Chars[6].width - Chars[7].x);
		f4 = abs(Chars[5].x + Chars[5].width - Chars[6].x);
		f5 = abs(Chars[4].x + Chars[4].width - Chars[5].x);
		// If the distance between the first and the second  character is bigger than the second
		 // and third , remove the first because it could be the number plate border
		if (abs(f2 - f1) >= 2) {
			Chars.erase(Chars.begin());

		}
		if (Chars.size() > 7) {

			if (abs(f3 - f4) > 3)
				Chars.pop_back();
		}

	}

	for (int i = 0; i < Chars.size(); i++) {            // for each contour

		// draw a green rect around the Chars
		cv::rectangle(result,                            // draw rectangle on original image
			Chars[i],        // Draw the rectagnle on the detected characters
			cv::Scalar(0, 255, 0),
			2);
		Detected_Chars_coordinations.push_back(Chars[i]);
		Mat ROI = Segment(Chars[i]);
		Chars_detected.push_back(ROI);

	}





}



//////////////////////   Charachter Recognition   //////////////////////

bool Characters_Recognition(vector<Rect>Detected_Chars_coordinations, vector<Mat> Chars_detected, String &Characters) {
	bool x;

	Mat matclassifications;                                                         //Declare a matrix for classifications
	FileStorage fsClassifications("Images&Classifications.xml", FileStorage::READ);  //Read from the Image and Classifications XML
	fsClassifications["classifications"] >> matclassifications;
	Mat matTrainingImagesAsFlattenedFloats;                                           //Declare a matrix for the Characters images
	fsClassifications["images"] >> matTrainingImagesAsFlattenedFloats;                  //Read from the Image and Classifications XML
	fsClassifications.release();

	cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();                              // Create the KNN algorithm

	kNearest->train(matTrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, matclassifications);      // Train the KNN from the XML file
	float fltCurrentChar;
	vector<String>Charss;

	for (int i = 0; i < Chars_detected.size(); i++) {

		Mat ROI;                                                        // Declare a martix for the detect chars
		ROI = Chars_detected[i];
		Mat matROIResized;
		matROIResized = ProcsessChar(ROI);                 // resize image, this will be more consistent for recognition and storage

		Mat matROIFloat;
		matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

		Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

		Mat matCurrentChar(0, 0, CV_32F);

		kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     //  find_nearest point when K=1

		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

		Characters = Characters + char(int(fltCurrentChar));

		Charss.push_back(Characters);


	}
	
	Characters = Characters;
	
	if (Charss.size() == 7) {
		cout << "Number Plate " << Characters << endl;
		x = true;
	}
	else if (Charss.size() >= 3 && Charss.size() != 7 && Charss.size() <= 9) { ////jj.size()>=1&&jj.size()!=7&&jj.size()<9
		cout << "Number Plate " << Characters << endl;



		cout << "Many Chararcters have been  recongize less or more than 7 please check the plate" << endl;

		x = true;
	}

	else { x = false; }


	return x;

}




///////// Verify the Character size///////////////////
bool verify_character_size(Mat Character) {

	//the Height a character in UK number plate is 70mm tall
	// and the Width is 50mm wide
	float aspectratio = 50.0f / 79.0f; //dimensions of a character
	float aspectratio1 = 14.0f / 79.0f; // dimensions of (1)


	//Compute the aspect ratio of the input character
	float charaspect = (float)Character.cols / (float)Character.rows;

	float error = 0.35; // set the error of the aspect 0.35
	float Max_Height = 90;  //set the Maximum height of the characters
	float Min_Height = 15;  // set the Minimum height of the characters

	//comptue the error of the aspect in the Maximum and the Minimum
	float Min_aspect = aspectratio1 - (aspectratio1*error);
	float Max_aspect = aspectratio + (aspectratio*error);
	// check if the input satisfied the paramerts
	return (charaspect > Min_aspect&&charaspect < Max_aspect&&Character.rows >= Min_Height && Character.rows < Max_Height);

}





////////////////////// Organize the characters based on thier location in the number plate //////////////////////
bool Hall(const Rect & a, const Rect & b) {
	return a.x < b.x;
}







///////// Process the verified characters to be same size as the training image in the KNN ///////////////
Mat ProcsessChar(Mat in) {
	int h = in.rows;
	int w = in.cols;
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = m / 2 - w / 2;
	transformMat.at<float>(1, 2) = m / 2 - h / 2;
	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
	Mat out;
	resize(warpImage, out, Size(20, 20));
	return out;
}




