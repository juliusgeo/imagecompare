#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

std::pair<int, int> extrema(float imslice[7][7]){
	int min=255;
	int max=0;
	int cur;
	for(int x=0;x<6;x++)
		{
			for (int y=0;y<6;y++)
			{
				cur=(int)imslice[x][y];
				if(cur<min){
					min=cur;
				}
				else if(cur>max){
					max=cur;
				}
			}
		}
	//std::cout << std::make_pair(min,max);
	return std::make_pair(min,max);
}
int is_keypoint(int cur, float imslice[7][7]){
	std::pair<int, int> ex=extrema(imslice);
	int min=ex.first;
	int max=ex.second;
	int a=(int)min-(int)max;
	if(abs(a)>11){
		if(cur == max || cur==min){
			return 1;
		}
	}
	return 0;

}
int main()
{
	cv::Mat imarray, blur1, blur2, fin;
	cv::Mat colorMat = cv::imread("lenna.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::cvtColor(colorMat, imarray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur( imarray, blur1,cv::Size( 3, 3 ), 0, 0 );
	cv::GaussianBlur( imarray, blur2, cv::Size( 9, 9 ), 0, 0 );
	cv::absdiff(blur1, blur2, fin);
	float imslice[7][7];
	int key;
	int end=0;
	long pixelNum=imarray.rows*imarray.cols;
	std::vector<cv::Point> coords;
	for(int j=3;j<fin.rows-3;j++){
		for (int i=3;i<fin.cols-3;i++){
			for(int x=-3;x<4;x++){
				for (int y=-3;y<4;y++){
					imslice[x][y]=(int)fin.at<uchar>(j+x,i+y);
					//std::cout<< imslice;
				}
			}
			key=is_keypoint((int)fin.at<uchar>(j,i), imslice);
			//std::cout<< key;
			if(key==1){
				//std::cout<< j << i;
				//coords[end][0]=j;
				//coords[end][1]=i;
				cv::Point coord=cv::Point(i,j);
				coords.push_back(coord);
				end++;
			}
		}
	}
	for(int z=0; z<end; z++){
		cv::circle(colorMat, coords[z], 1,cv::Scalar(255,255,255), CV_FILLED);
	}
	//cv::imshow("After",imarray);
	//cv::waitKey(2000);
	cv::imwrite("out.jpg", colorMat );
  	return 0;
}
