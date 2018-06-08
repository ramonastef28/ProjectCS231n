/*
The MIT License

Copyright (c) 2018 Ramona Stefanescu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>
#include <string.h>
#include <tf/transform_datatypes.h>
#include <unistd.h>
#include <iostream>
#include "lev_localization/LocationalImage.h"
#include "vo.h"

//#define PUBLISH_IMAGE

std::ofstream debug_out("z.txt");

/////////////////// forward declaration
void init_vo();
void do_vo();
double get_scale();
void featureTracking(Mat& img_1, Mat& img_2, std::vector<Point2f>& points1, std::vector<Point2f>& points2, std::vector<uchar>& status);
void orb_feature_detection(Mat&, std::vector<Point2f>&);
void fast_feature_detection(Mat&, std::vector<Point2f>&);

////////////////// define operation function pointer
void (*vo_func)();
void (*feature_detection)(Mat&, std::vector<Point2f>&);

///////////////// constant
const double focal = 1386.322107505445;
const cv::Point2d pp(989.8917867402782, 595.3123933850414);

//////////////// variables
int n_data = 0;
cv_bridge::CvImagePtr prev_img, curr_img, prev_src, prev_undst, curr_src, curr_undst;
Mat E, R, t, mask;
Mat R_f, t_f;
std::vector<Point2f> prev_features, curr_features;
geometry_msgs::Pose prev_pose, curr_pose;
float roll, pitch, yaw;
float vo_roll, vo_pitch, vo_yaw;
float* vo_euler;
FILE *VO;

////pre-process variables
float m[] = {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, -476, 24, 6, 4, 16, 24, 1, 4, 1, 4, 6, 4, 1};
cv::Mat kernel;
Point anchor = Point(-1, -1);
float delta = 0;
int ddepth = -1;
float cm[] = {1391.6, 0.0, 968.0230, 0.0, 1395.9, 587.5922, 0.0, 0.0, 1.0};
cv::Mat camera_matrix{3, 3, CV_32F, cm};
float d[] = {-0.1778, 0.1012, -0.0009995, 0.0007424, 0.0};
cv::Mat dist_coefs = Mat(1, 5, CV_32F, d);

ros::Publisher odom_pub;
ros::Publisher image_pub;

#ifdef PLOT
Mat traj = Mat::zeros(600, 600, CV_8UC3);
char text[100];
int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 1;
int thickness = 1;
cv::Point textOrg(10, 50);
#endif

bool isRotationMatrix(Mat& R_f) {
  Mat Rt;
  transpose(R_f, Rt);
  Mat shouldBeIdentity = Rt * R_f;
  Mat I = Mat::eye(3, 3, shouldBeIdentity.type());
  return norm(I, shouldBeIdentity) < 1e-6;
}

void rotationMatrixToEulerAngles(Mat& R_f, float* euler)

{
  assert(isRotationMatrix(R_f));
  float sy = sqrt(R_f.at<double>(0, 0) * R_f.at<double>(0, 0) + R_f.at<double>(1, 0) * R_f.at<double>(1, 0));
  bool singular = sy < 1e-6;
  if (!singular) {
    euler[0] = atan2(R_f.at<double>(2, 1), R_f.at<double>(2, 2));
    euler[1] = atan2(-R_f.at<double>(2, 0), sy);
    euler[2] = atan2(R_f.at<double>(1, 0), R_f.at<double>(0, 0));
  } else {
    euler[0] = atan2(-R_f.at<double>(1, 2), R_f.at<double>(1, 1));
    euler[1] = atan2(-R_f.at<double>(2, 0), sy);
    euler[2] = 0;
  }
}

void QuaternionToEulerAngles() {
  double sinr = +2.0 * (curr_pose.orientation.w * curr_pose.orientation.x + curr_pose.orientation.y * curr_pose.orientation.z);
  double cosr = +1.0 - 2.0 * (curr_pose.orientation.x * curr_pose.orientation.x + curr_pose.orientation.y * curr_pose.orientation.y);
  roll = atan2(sinr, cosr);
  // pitch (y-axis rotation)
  double sinp = +2.0 * (curr_pose.orientation.w * curr_pose.orientation.y - curr_pose.orientation.z * curr_pose.orientation.x);
  if (fabs(sinp) >= 1)
    pitch = copysign(M_PI / 2, sinp);  // use 90 degrees if out of range
  else
    pitch = asin(sinp);
  // yaw (z-axis rotation)
  double siny = +2.0 * (curr_pose.orientation.w * curr_pose.orientation.z + curr_pose.orientation.x * curr_pose.orientation.y);
  double cosy = +1.0 - 2.0 * (curr_pose.orientation.y * curr_pose.orientation.y + curr_pose.orientation.z * curr_pose.orientation.z);
  yaw = atan2(siny, cosy);
}

double get_scale() {
  return sqrt((curr_pose.position.x - prev_pose.position.x) * (curr_pose.position.x - prev_pose.position.x) +
              (curr_pose.position.y - prev_pose.position.y) * (curr_pose.position.y - prev_pose.position.y) +
              (curr_pose.position.z - prev_pose.position.z) * (curr_pose.position.z - prev_pose.position.z));
}

void init_vo() {
  std::vector<uchar> status;
  feature_detection(prev_img->image, prev_features);
  featureTracking(prev_img->image, curr_img->image, prev_features, curr_features, status);  // track those features to current image

  E = findEssentialMat(curr_features, prev_features, focal, pp, RANSAC, 0.999, RANSAC_THRESHOLD, mask);
  recoverPose(E, curr_features, prev_features, R, t, focal, pp, mask);

  R_f = R.clone();
  t_f = t.clone();

  prev_features = curr_features;
  prev_img = curr_img;
  prev_pose = curr_pose;
}

void do_vo() {
  std::vector<uchar> status;
  featureTracking(prev_img->image, curr_img->image, prev_features, curr_features, status);

  E = findEssentialMat(curr_features, prev_features, focal, pp, RANSAC, 0.999, RANSAC_THRESHOLD, mask);
  recoverPose(E, curr_features, prev_features, R, t, focal, pp, mask);

  double scale = get_scale();
  float vo_euler[3];
  if ((scale > 0.3) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
    t_f = t_f + scale * (R_f * t);
    R_f = R * R_f;
  } else {
    ROS_WARN("scale below 0.3, or incorrect translation %f,[%f,%f,%f]", scale, t.at<double>(0), t.at<double>(1), t.at<double>(2));
  }

  ROS_INFO("pose %d %.3f %.3f %.3f", n_data, t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
  rotationMatrixToEulerAngles(R_f, vo_euler);
  // fprintf(VO,"%f %f %f\n",t_f.at<double>(0), t_f.at<double>(2), *vo_euler);
  tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, *vo_euler);                        // Create this quaternion from roll/pitch/yaw (in radians)
  ROS_INFO_STREAM("Quaternions: " << q[0] << " " << q[1] << " " << q[2] << " " << q[3]);  // Print the quaternion components (0,0,0,1)
  QuaternionToEulerAngles();
  nav_msgs::Odometry odom;
  odom.header.frame_id = "base_link";
  odom.child_frame_id = "camera_vo";
  odom.pose.pose.position.x = t_f.at<double>(0);
  odom.pose.pose.position.z = t_f.at<double>(1);
  odom.pose.pose.position.y = t_f.at<double>(2);
  odom.pose.pose.orientation.x = q[0];
  odom.pose.pose.orientation.y = q[1];
  odom.pose.pose.orientation.z = q[2];
  odom.pose.pose.orientation.w = q[3];
  odom_pub.publish(odom);

  debug_out << odom.pose.pose.position.x << "," << odom.pose.pose.position.y << "," << odom.pose.pose.position.z << std::endl;

  // perform feature detection if # features falls below a threshold
  if (prev_features.size() < MIN_NUM_FEAT) {
    ROS_INFO("feature re-detection, previous feature count: %d", (int)(prev_features.size()));
    feature_detection(prev_img->image, prev_features);
    featureTracking(prev_img->image, curr_img->image, prev_features, curr_features, status);
  }

#ifdef PUBLISH_IMAGE
  static int img_seq = 0;
  // convert features to keypoints and plot it
  std::vector<KeyPoint> kpts;
  KeyPoint::convert(curr_features, kpts);
  Mat keypoints_img;
  drawKeypoints(curr_img->image, kpts, keypoints_img, CV_RGB(0, 0, 255));
  sensor_msgs::Image img_out;

  img_out.header.stamp = ros::Time::now();
  img_out.header.seq = img_seq++;
  img_out.header.frame_id = "";
  img_out.height = keypoints_img.rows;
  img_out.width = keypoints_img.cols;
  img_out.step = *(keypoints_img.step.p);
  img_out.encoding = sensor_msgs::image_encodings::RGB8;
  img_out.data.resize(img_out.height * img_out.step);
  memcpy(&img_out.data[0], keypoints_img.data, img_out.data.size());
  image_pub.publish(img_out);

#endif

  prev_features = curr_features;
  prev_img = curr_img;
  prev_pose = curr_pose;

#ifdef PLOT
  int x = int(t_f.at<double>(0)) + 300;
  int y = int(t_f.at<double>(2)) + 250;
  circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);
  rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
  sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
  putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

  imshow("front_camera", curr_img->image);
  imshow("trajectory", traj);
  waitKey(1);
#endif
}

void on_data(const lev_localization::LocationalImageConstPtr& msg) {
  switch (n_data) {
    case 0:
      prev_img = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::MONO8);
      // undistort(prev_src->image, prev_undst->image,camera_matrix, dist_coefs);
      // filter2D(prev_undst->image, prev_img->image,ddepth,kernel,anchor,delta,cv::BORDER_DEFAULT);
      break;
    case 1:
      curr_img = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::MONO8);
      // undistort(curr_src->image, curr_undst->image, camera_matrix, dist_coefs);
      // filter2D(curr_undst->image, curr_img->image,ddepth,kernel,anchor,delta,cv::BORDER_DEFAULT);
      curr_pose = msg->odom.pose.pose;
      vo_func = init_vo;
      break;
    default:
      curr_img = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::MONO8);
      // undistort(curr_src->image, curr_undst->image, camera_matrix, dist_coefs);
      // filter2D(curr_undst->image, curr_img->image,ddepth,kernel,anchor,delta,cv::BORDER_DEFAULT);
      curr_pose = msg->odom.pose.pose;
      vo_func = do_vo;
  }
  if (vo_func) vo_func();
  n_data++;
}

int main(int argc, char** argv) {
  // VO = fopen("VO.txt","a");
  kernel = Mat(5, 5, CV_32F, m);
  kernel = cv::Mat((-0.5f / 120.4f) * kernel);

  ros::init(argc, argv, "rtvo");

#ifdef PLOT
  namedWindow("trajectory", WINDOW_AUTOSIZE);
  namedWindow("front_camera", WINDOW_AUTOSIZE);
#endif
  // parse options
  int c, fflag = 0;
  feature_detection = orb_feature_detection;
  char* fname;
  while ((c = getopt(argc, argv, "f:")) != -1) {
    switch (c) {
      case 'f':
        fflag = 1;
        fname = optarg;
        if (strncmp(fname, "fast", strlen("fast")) == 0) {
          ROS_INFO("use FAST feature detection");
          feature_detection = fast_feature_detection;
        } else {
          ROS_INFO("unknown feature detection algorithm %s, default to ORB detection", fname);
        }
    }
  }

  if (!fflag) {
    ROS_INFO("default to ORB detection");
  }

  vo_func = 0;

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe(SYNC_TOPIC, 100, on_data);
  odom_pub = n.advertise<nav_msgs::Odometry>("/vo/mono/odom", 50);
  image_pub = n.advertise<sensor_msgs::Image>("/vo/mono/image", 10);
  ROS_INFO("starting realtime video odometry...");
  ros::spin();
  return 0;
}

void featureTracking(Mat& img_1, Mat& img_2, std::vector<Point2f>& points1, std::vector<Point2f>& points2, std::vector<uchar>& status) {
  // this function automatically gets rid of points for which tracking fails
  std::vector<float> err;
  Size winSize = Size(21, 21);
  TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  // getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for (size_t i = 0; i < status.size(); i++) {
    Point2f pt = points2.at(i - indexCorrection);
    if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
      if ((pt.x < 0) || (pt.y < 0)) {
        status.at(i) = 0;
      }
      points1.erase(points1.begin() + (i - indexCorrection));
      points2.erase(points2.begin() + (i - indexCorrection));
      indexCorrection++;
    }
  }
}

void fast_feature_detection(Mat& img_1, std::vector<Point2f>& points1) {  // uses FAST as of now, modify parameters as necessary
  std::vector<KeyPoint> keypoints_1;
  int fast_threshold = FAST_THRESHOLD;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, std::vector<int>());
}

void orb_feature_detection(Mat& img_1, std::vector<Point2f>& points1) {
  std::vector<KeyPoint> keypoints_1;

  int nfeatures = 2000;
  float scaleFactor = 1.2f;
  int nlevels = 8;
  int edgeThreshold = 31;  // Changed default (31);
  int firstLevel = 0;
  int WTA_K = 2;
  int scoreType = ORB::FAST_SCORE;
  int patchSize = 31;
  int fastThreshold = FAST_THRESHOLD;
  Ptr<ORB> detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  detector->detect(img_1, keypoints_1);
  Mat orbimg;
  int flags = 2;
  drawKeypoints(img_1, keypoints_1, orbimg, CV_RGB(0, 0, 255), flags);
  KeyPoint::convert(keypoints_1, points1, std::vector<int>());
}
