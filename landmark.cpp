#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#define PI 3.14159265
 
 
using namespace std;
using namespace cv;
using namespace cv::face;


string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

Point2f get_left_eye_centroid(const vector<Point2f>  landmark){
  Point2f center;
  float x, y;
  for (int i = 36; i <= 41; ++i)
  {
    x += landmark[i].x;
    y += landmark[i].y;
  }

  center.x = x / 6.0;
  center.y = y / 6.0;

  return center;
}


Point2f get_right_eye_centroid(const vector<Point2f>  landmark){
  Point2f center;
  float x, y;
  for (int i = 42; i <= 47; ++i)
  {
    x += landmark[i].x;
    y += landmark[i].y;
  }

  center.x = x / 6.0;
  center.y = y / 6.0;

  return center;
}


float get_rotate_face(const std::vector<Point2f> landmark){

    float distance = 0;
    Point2f vector_eyes = Point2f(landmark[29].x - landmark[27].x, landmark[29].y-landmark[27].y);
    float denominator = sqrt(vector_eyes.x*vector_eyes.x + vector_eyes.y*vector_eyes.y);

    Point2f coor_mid_eyes = landmark[27];
    
    for (int i = 27; i <= 35; ++i)
    {
      // calculated eviation nose to eyes axis line
      distance += (vector_eyes.y*landmark[i].x - vector_eyes.x*landmark[i].y 
              - vector_eyes.y*coor_mid_eyes.x + vector_eyes.x*coor_mid_eyes.y) / denominator;
    }
    return distance;
}


float get_tilted_face(const Point2f& vector_1, const Point2f& vector_2){

  float numerator = vector_1.x*vector_2.x + vector_1.y*vector_2.y;

  float denominator =  sqrt(vector_1.x*vector_1.x + vector_1.y*vector_1.y)
                      *sqrt(vector_2.x*vector_2.x + vector_2.y*vector_2.y);

  float ratio = numerator/denominator;

  float angle =  acos(ratio)* 180.0 / PI;;

  return angle;
}




int main(int argc,char** argv)
{
    // Load Face Detector
    CascadeClassifier faceDetector("haarcascade_frontalface_alt.xml");
 
    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();
 
    // Load landmark detector
    facemark->loadModel("lbfmodel.yaml");
 
    // Set up webcam for video capture
    VideoCapture cam(0);
     
    // Variable to store a video frame and its grayscale 
    Mat frame, gray;

    Point2f leftEye;
    Point2f rightEye;
	namedWindow("src");
	namedWindow("left_ear");
	namedWindow("right_ear");
	namedWindow("nose");
	namedWindow("left_eye");
	namedWindow("right_eye");
	namedWindow("left_eye_brown");
	namedWindow("right_eye_brown");
	namedWindow("mouth");


    
    bool check_tilted = false;
    bool check_rotate = false;


    // Read a frame
    while(cam.read(frame))
    {
       
      // cv::flip(frame, frame, 1); 
      // frame = cv::imread("3.jpg", IMREAD_COLOR);

      // Find face
      vector<Rect> faces;
      // Convert frame to grayscale because
      // faceDetector requires grayscale image.
      cvtColor(frame, gray, COLOR_BGR2GRAY);
 		
      // Detect faces
      faceDetector.detectMultiScale(gray, faces);
       
      // Variable for landmarks. 
      // Landmarks for one face is a vector of points
      // There can be more than one face in the image. Hence, we 
      // use a vector of vector of points. 
      vector< vector<Point2f> > landmarks;

      // Feature on face
      cv::Mat nose;
      cv::Mat left_eye;
      cv::Mat right_eye;
      cv::Mat mouth;
      cv::Mat left_ear;
      cv::Mat right_ear;
      cv::Mat left_eye_brown;
      cv::Mat right_eye_brown;
      cv::Mat tmp = frame.clone();

      // Run landmark detector
      bool success = facemark->fit(frame,faces,landmarks);

      if(success)
      {
      	check_tilted = false;
      	check_rotate = false;

        // If successful, render the landmarks on the face
        for(int i = 0; i < 1; i++)
        {
          left_ear = tmp(cv::Rect(cv::Point(landmarks[i][0].x-20, landmarks[i][0].y-10), cv::Point(landmarks[i][2].x, landmarks[i][2].y)));
          right_ear = tmp(cv::Rect(cv::Point(landmarks[i][16].x, landmarks[i][16].y-10), cv::Point(landmarks[i][14].x+20, landmarks[i][14].y)));
          nose = tmp(cv::Rect(cv::Point(landmarks[i][31].x-5, landmarks[i][27].y), cv::Point(landmarks[i][35].x+5, landmarks[i][33].y)));
          left_eye = tmp(cv::Rect(cv::Point(landmarks[i][36].x-10, landmarks[i][37].y-10), cv::Point(landmarks[i][39].x+10, landmarks[i][41].y+10)));
          right_eye = tmp(cv::Rect(cv::Point(landmarks[i][42].x-10, landmarks[i][44].y-10), cv::Point(landmarks[i][45].x+10, landmarks[i][46].y+10)));
          left_eye_brown = tmp(cv::Rect(cv::Point(landmarks[i][17].x, landmarks[i][19].y-10), cv::Point(landmarks[i][21].x, landmarks[i][21].y+10)));
          right_eye_brown = tmp(cv::Rect(cv::Point(landmarks[i][22].x, landmarks[i][24].y-10), cv::Point(landmarks[i][26].x, landmarks[i][26].y+10)));
          mouth = tmp(cv::Rect(cv::Point(landmarks[i][48].x, landmarks[i][50].y-5), cv::Point(landmarks[i][54].x, landmarks[i][57].y+5)));
          // face::drawFacemarks(frame, landmarks[i], Scalar(0,0,255));

          leftEye  = get_left_eye_centroid(landmarks[i]);
          rightEye = get_right_eye_centroid(landmarks[i]);

          Point2f vector_eyes = Point2f(rightEye.x-leftEye.x, rightEye.y-leftEye.y);

          Point2f center_Eye  = Point(int((leftEye.x+rightEye.x)/2.0), int((leftEye.y+rightEye.y)/2.0));

          
          float tilted_angle = get_tilted_face(Point2f(1,0), vector_eyes);
          
          if(tilted_angle > 4 && leftEye.y - rightEye.y > 0){
            putText(frame, " Left", Point(300, 50), FONT_HERSHEY_PLAIN, 2,  Scalar(255,0,0));
          }
          else if(tilted_angle > 4 && leftEye.y - rightEye.y < 0){
            putText(frame, " Right", Point(300, 50), FONT_HERSHEY_PLAIN, 2,  Scalar(255,0,0));            
          }
          else{
          	check_tilted = true;
          }

          putText(frame, "Tiled angle:" +intToString(int(tilted_angle)), Point(50, 50), FONT_HERSHEY_PLAIN, 2,  Scalar(255,0,0));

          // cout<<get_rotate_face(landmarks[i])<<endl;
          circle(frame, leftEye, 3, Scalar(0,0,255), FILLED);
          circle(frame, rightEye, 3, Scalar(0,0,255), FILLED);
          circle(frame, center_Eye , 3, Scalar(0,0,255), FILLED);


          Point p1(0,0);
          Point p2(0,0);
          // cout<<get_rotate_face(landmarks[i])<<endl;
          float rotate_value = get_rotate_face(landmarks[i]);

          if ( abs(rotate_value) <= 15 ){
            p1 = Point(int(center_Eye.x), int(center_Eye.y));
            p2 = Point(int(center_Eye.x), int(center_Eye.y));
          	putText(frame,"No Rotate", Point(50, 80), FONT_HERSHEY_PLAIN, 2,  Scalar(255,0,0));
          	check_rotate = true;
          }
          else if(rotate_value > 15){
            p1 = Point(int(center_Eye.x), int(center_Eye.y));
            p2 = Point(int(center_Eye.x-rotate_value), int(center_Eye.y));
          	putText(frame,"Rotate Left" , Point(50, 80), FONT_HERSHEY_PLAIN, 2,  Scalar(255,0,0));
          }
          else{
            p1 = Point(int(center_Eye.x), int(center_Eye.y));
            p2 = Point(int(center_Eye.x-rotate_value), int(center_Eye.y));            
          	putText(frame,"Rotate Right", Point(50, 80), FONT_HERSHEY_PLAIN, 2,  Scalar(255,0,0));
          }

          arrowedLine(frame, p1, p2, Scalar(0, 255, 0),3);          
          putText(frame, intToString(int(rotate_value)), p1, FONT_HERSHEY_PLAIN, 2,  Scalar(255,0,0));

        }
      }else{
          putText(frame,"Can't detect face ", Point(50, 80), FONT_HERSHEY_PLAIN, 2,  Scalar(0,0,255));      	
      }

	 
      // Display results 
      imshow("src", frame);

      	if(check_tilted && check_rotate){  	
	      	imshow("left_ear", left_ear);
	      	imshow("right_ear", right_ear);
	      	imshow("nose", nose);
	      	imshow("left_eye", left_eye);
	      	imshow("right_eye", right_eye);
	      	imshow("left_eye_brown", left_eye_brown);
	      	imshow("right_eye_brown", right_eye_brown);
	      	imshow("mouth", mouth);
 		}
 		else{
 			nose = cv::Mat::zeros(Size(100,100), CV_8UC3);
      		left_eye =cv::Mat::zeros(Size(100,100), CV_8UC3);
      		right_eye=cv::Mat::zeros(Size(100,100), CV_8UC3);
      		mouth=cv::Mat::zeros(Size(100,100), CV_8UC3);
      		left_ear=cv::Mat::zeros(Size(100,100), CV_8UC3);
      		right_ear=cv::Mat::zeros(Size(100,100), CV_8UC3);
      		left_eye_brown=cv::Mat::zeros(Size(100,100), CV_8UC3);
      		right_eye_brown=cv::Mat::zeros(Size(100,100), CV_8UC3);

	      	imshow("left_ear", left_ear);
	      	imshow("right_ear", right_ear);
	      	imshow("nose", nose);
	      	imshow("left_eye", left_eye);
	      	imshow("right_eye", right_eye);
	      	imshow("left_eye_brown", left_eye_brown);
	      	imshow("right_eye_brown", right_eye_brown);
	      	imshow("mouth", mouth); 			
 		}
 		// waitKey(0);
      	// Exit loop if ESC is pressed
      	if (waitKey(1) == 27) break;
    }
    return 0;
}
