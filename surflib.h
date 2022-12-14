#ifndef SURFLIB_H
#define SURFLIB_H

#include <opencv2/opencv.hpp>

#include "integral.h"
#include "fasthessian.h"
#include "surf.h"
#include "ipoint.h"
#include "utils.h"


//! Library function builds vector of described interest points
inline void surfDetDes(cv::Mat& img,  /* image to find Ipoints in */
	std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
	bool upright = true, /* run in rotation invariant mode? */
	int octaves = OCTAVES, /* number of octaves to calculate */
	int intervals = INTERVALS, /* number of intervals per octave */
	int init_sample = INIT_SAMPLE, /* initial sampling step */
	float thres = THRES /* blob response threshold */)
{
	// Create integral-image representation of the image
	cv::Mat int_img;// = ;
	Integral(img, int_img);

	// Create Fast Hessian Object
	FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

	// Extract interest points and store in vector ipts
	fh.getIpoints();

	// Create Surf Descriptor Object
	Surf des(int_img, ipts);

	// Extract the descriptors for the ipts
	des.getDescriptors(upright);

}


//! Library function builds vector of interest points
inline void surfDet(cv::Mat& img,  /* image to find Ipoints in */
	std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
	int octaves = OCTAVES, /* number of octaves to calculate */
	int intervals = INTERVALS, /* number of intervals per octave */
	int init_sample = INIT_SAMPLE, /* initial sampling step */
	float thres = THRES /* blob response threshold */)
{
	// Create integral image representation of the image
	cv::Mat int_img;// =
	Integral(img, int_img);

	// Create Fast Hessian Object
	FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

	// Extract interest points and store in vector ipts
	fh.getIpoints();

}




//! Library function describes interest points in vector
inline void surfDes(cv::Mat& img,  /* image to find Ipoints in */
	std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
	bool upright = false) /* run in rotation invariant mode? */
{
	// Create integral image representation of the image
	cv::Mat int_img;// =
	Integral(img, int_img);

	// Create Surf Descriptor Object
	Surf des(int_img, ipts);

	// Extract the descriptors for the ipts
	des.getDescriptors(upright);
}


#endif
