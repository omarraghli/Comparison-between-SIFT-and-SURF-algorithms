#include "surflib.h"
#include "kmeans.h"
#include <ctime>
#include <iostream>
#include <ctime>
#include "sift.h"
//-------------------------------------------------------
// Define PROCEDURE as:
//  - 1 and supply image path to run on static image
//  - 2 to capture from a webcam
//  - 3 to match find an object in an image (work in progress)
//  - 4 to display moving features (work in progress)
//  - 5 to show matches between static images
#define PROCEDURE 1
#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world400d.lib")
#else
#pragma comment(lib, "opnecv_world400.lib")
#endif
//-------------------------------------------------------

int mainImage(void)
{
	// Declare Ipoints and other stuff
	IpVec ipts;
	cv::Mat img= cv::imread(".\\imgs\\sf.jpg", cv::IMREAD_COLOR);

	// Detect and describe interest points in the image
	clock_t start = clock();
	surfDetDes(img, ipts, false, 5, 4, 2, 0.0004f);
	clock_t end = clock();

	std::cout<< "OpenSURF found: " << ipts.size() << " interest points" << std::endl;
	std::cout<< "OpenSURF took: " << float(end - start) / CLOCKS_PER_SEC  << " seconds" << std::endl;

	// Draw the detected points
	drawIpoints(img, ipts);

	// Display the result
	showImage(img);

	return 0;
}

//-------------------------------------------------------




//-------------------------------------------------------


//-------------------------------------------------------


//-------------------------------------------------------

int mainStaticMatch()
{

    //FilePicker for the first image
    FILE *in;
    if (!(in = popen("zenity  --title=\"Select an image\" --file-selection","r"))){return 1;}
    char buff[512];
    std::string selectFile = "";
    while (fgets(buff, sizeof(buff), in) != NULL) {
        selectFile += buff;
    }
    pclose(in);
    selectFile.erase(std::remove(selectFile.begin(), selectFile.end(), '\n'),selectFile.end());
    cv::Mat img1 = cv::imread(selectFile);


    //FilePicker for the secound image
    FILE *in2;
    if (!(in2 = popen("zenity  --title=\"Select an image\" --file-selection","r"))){return 1;}
    char buff2[512];
    std::string selectFile2 = "";
    while (fgets(buff2, sizeof(buff2), in2) != NULL) {
        selectFile2 += buff2;
    }
    pclose(in2);

    selectFile2.erase(std::remove(selectFile2.begin(), selectFile2.end(), '\n'),selectFile2.end());
    cv::Mat img2 = cv::imread(selectFile2);


    //------------------------------------------------------SIFT ALGORITHM IMPLEMENTATION-------------------------------------------------------------------------
    //calculer le temp d'éxecution
    clock_t temps_initial, temps_final, temps_initial_surf, temps_final_surf  ;
    float temps_cpu;
    vector<extrema> ex1, ex2;

    Mat image, colorfulImage;
    unsigned n_pixels1, n_pixels2;
    colorfulImage = img1;
    rescale(colorfulImage, image);
    temps_initial = clock ();
    ex1 = SIFTDescript(image, n_pixels1);
    //cout << "From " << n_pixels1 << " point, ";
    //cout << "we extracted " << ex1.size() << " interest points." << endl;


    Mat image2, colorfulImage2;
    colorfulImage2 = img2;
    rescale(colorfulImage2, image2);

    ex2 = SIFTDescript(image2, n_pixels2);


    //cout << "From " << n_pixels2 << " point, ";
    //cout << "we extracted " << ex2.size() << " interest points." << endl;

    for(auto e:ex2){
        circle(colorfulImage2, e.pt, 3, Scalar(0,255,255));
    }

    for(const auto& e:ex1){
        circle(colorfulImage, e.pt, 3, Scalar(0,255,255));
    }
    /*imshow("show", colorfulImage);
    waitKey(0);
    imshow("show", colorfulImage2);
    waitKey(0);*/

    vector<Point> matchKP1, matchKP2;
    unsigned cnt_match = 0;
    findMatches(ex1, ex2, matchKP1, matchKP2, cnt_match);
    temps_final  = clock ();
    temps_cpu = (float)(temps_final - temps_initial) / CLOCKS_PER_SEC ; // millisecondes
    cout << "execution time for SIFT is: " << temps_cpu <<" Sec" << endl;
    cout << "SIFT Matches: " << cnt_match << endl;


    // Show SIFT matches
    int height = colorfulImage.rows;
    if(colorfulImage.rows < colorfulImage2.rows) height = colorfulImage2.rows;
     Mat matchesSIFT(height, colorfulImage.cols + colorfulImage2.cols, CV_8UC3);
    for(int r = 0; r < colorfulImage.rows; r++){
        for(int c = 0; c < colorfulImage.cols; c++)
            matchesSIFT.at<Vec3b>(r,c) = colorfulImage.at<Vec3b>(r,c);
    }
    for(int r = 0; r < colorfulImage2.rows; r++){
        for(int c = 0; c < colorfulImage2.cols; c++)
            matchesSIFT.at<Vec3b>(r,c+colorfulImage.cols) = colorfulImage2.at<Vec3b>(r,c);
    }
    for(int i=0; i<matchKP1.size(); i++){
        line(matchesSIFT,matchKP1[i],Point(matchKP2[i].x+colorfulImage.cols, matchKP2[i].y),Scalar(200,200,0));
    }
    imshow("SIFTMatches", matchesSIFT);


//------------------------------------------------------SURF ALGORITHM IMPLEMENTATION-----------------------------------------------------

    float temps_cpu_surf;

    Mat image11,image22;
    IpVec ipts1, ipts2;

    rescale(img1, image11);
    rescale(img2, image22);
    //calculer le temp d'éxecution
    temps_initial_surf = clock ();

	surfDetDes(image11,ipts1,false,4,4,2,0.0001f);
	surfDetDes(image22,ipts2,false,4,4,2,0.0001f);

	IpPairVec matches;
	getMatches(ipts1,ipts2,matches);

    temps_final_surf  = clock ();
    temps_cpu_surf = (float)(temps_final_surf - temps_initial_surf) / CLOCKS_PER_SEC; // millisecondes
    cout << "execution time for SURF is: " << temps_cpu_surf <<" Sec" << endl;

	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		drawPoint(img1,matches[i].first);
		drawPoint(img2,matches[i].second);

		const int & w = img1.cols;
		cv::line(img1,cv::Point(matches[i].first.x,matches[i].first.y),cv::Point(matches[i].second.x+w,matches[i].second.y), cv::Scalar(255,255,255),1);
		cv::line(img2,cv::Point(matches[i].first.x-w,matches[i].first.y),cv::Point(matches[i].second.x,matches[i].second.y), cv::Scalar(255,255,255),1);
	}

	std::cout<< "SURF Matches: " << matches.size();




    // Show SURF matches

    int heightSURF = img1.rows;
    if(img1.rows < img2.rows) heightSURF = img2.rows;
     Mat matchesSURF(heightSURF, img1.cols + img2.cols, CV_8UC3);
    for(int r = 0; r < img1.rows; r++){
        for(int c = 0; c < img1.cols; c++)
            matchesSURF.at<Vec3b>(r,c) = img1.at<Vec3b>(r,c);
    }
    for(int r = 0; r < img2.rows; r++){
        for(int c = 0; c < img2.cols; c++)
            matchesSURF.at<Vec3b>(r,c+img1.cols) = img2.at<Vec3b>(r,c);
    }
    /*for(int i=0; i<matchKP1.size(); i++){
        line(matchesSURF,matchKP1[i],Point(matchKP2[i].x+img1.cols, matchKP2[i].y),Scalar(200,200,0));
    }*/
    imshow("SURFMatches", matchesSURF);




	//cv::namedWindow("SURF1");
	//cv::namedWindow("SURF2");
	//cv::imshow("SURF1", img1);
	//cv::imshow("SURF2",img2);
	cv::waitKey(0);

	return 0;
}

//-------------------------------------------------------


//-------------------------------------------------------

int main(void)
{
//////////////////////////////////////////////






////////////////////////////////////
	if( PROCEDURE == 2)
		mainImage();
	else
		mainStaticMatch();
}
