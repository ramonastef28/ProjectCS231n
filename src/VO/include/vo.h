#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;

//#define PLOT
#define RANSAC_THRESHOLD 1.5
// FIXME: configurable
#define FAST_THRESHOLD 50
#define SYNC_TOPIC "/loc_sync_data"

#define MIN_NUM_FEAT 2000
