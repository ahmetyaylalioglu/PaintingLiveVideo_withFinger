/*
Ahmet Yaylalioglu
23/04/2017
this is non commercial opencv application
This application allows drawing with different colors into live webcam video with only human finger (not another objects)
This project include Mat to IplImage and IplImage to Mat conversion implementation
Thanks to Philipp Wagner <bytefish[at]gmx[dot]de> for skin color detection implementation
and thanks to color palet idea and color palet image Farshid Tavakolizadeh
1)Skin Color Detection
2)ConvexHull for finger detection
3)Thresholding and Drawing
*/

#include <opencv\cv.h>
#include <opencv\highgui.h>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using std::cout;
using std::endl;
CvCapture* cap = 0;
IplImage* Ana_goruntu;
Mat deri, esikGoruntu, fgMaskMOG2;

#define mavi CV_RGB(0,0,255)
#define yesil CV_RGB(0,255,0)
#define kirmizi CV_RGB(255,0,0)
#define beyaz CV_RGB(255,255,255)
#define siyah CV_RGB(0,0,0)


void EkraniTemizle(IplImage* imgKarala, IplImage* imgSayfa){
	cvSet(imgKarala, siyah);
	cvSet(imgSayfa, beyaz);
}


bool R1(int R, int G, int B) {
	bool e1 = (R>95) && (G>40) && (B>20) && ((max(R, max(G, B)) - min(R, min(G, B)))>15) && (abs(R - G)>15) && (R>G) && (R>B);
	bool e2 = (R>220) && (G>210) && (B>170) && (abs(R - G) <= 15) && (R>B) && (G>B);
	return (e1 || e2);
}

bool R2(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862*Cb + 20;
	bool e4 = Cr >= 0.3448*Cb + 76.2069;
	bool e5 = Cr >= -4.5652*Cb + 234.5652;
	bool e6 = Cr <= -1.15*Cb + 301.75;
	bool e7 = Cr <= -2.2857*Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
	return (H<25) || (H > 230);
}

Mat DeriRengi(Mat const &src) {
	Mat dst = src.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);

	Mat src_ycrcb, src_hsv;
	cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
	src.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			
			bool a = R1(R, G, B);

			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			
			bool b = R2(Y, Cr, Cb);

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			
			bool c = R3(H, S, V);

			if (!(a&&b&&c))
				dst.ptr<Vec3b>(i)[j] = cblack;
		}
	}
	return dst;
}




int main() {

	Ptr<BackgroundSubtractor> pMOG2;
	pMOG2 = new BackgroundSubtractorMOG2();
	IplImage* imgKarala = NULL; //elle cizdiðimiz veriyi tutan nesne,elimizin pozisyonlarýnýn takip edildiði nesne
	cap = cvCaptureFromCAM(0);
	if (!cap){
		cout << "KAMERA HATASI" << endl;
		return -1;
	}

	IplImage* imgRenkPaleti = 0;
	imgRenkPaleti = cvLoadImage("palet.panel", CV_LOAD_IMAGE_COLOR);
	if (!imgRenkPaleti){
		cout << "Panel Dosyasi bulunamadi" << endl;
		return -1;
	}
	IplImage* imgSayfa = 0;
	imgSayfa = cvCreateImage(cvSize(cvQueryFrame(cap)->width, cvQueryFrame(cap)->height),cvQueryFrame(cap)->depth,3);
	cvSet(imgSayfa, beyaz);

	CvFont yazi, buyukYazi;
	cvInitFont(&yazi, CV_FONT_HERSHEY_COMPLEX, 1, .6, 0, 2, CV_AA);
	cvInitFont(&buyukYazi, CV_FONT_HERSHEY_COMPLEX, 3, .6, 0, 3, CV_AA);
	int kapatma_sayac = 10, temizle_sayac = 20; //kapatma ve temizleme islemleri icin sayaclar
	char yazidizi[50];//yazilari tutmak için buffer
	int goruntu_kayit_no = 0;
	int pozisyonX = 0;
	int pozisyonY = 0;
	double alan_limit = 700;
	int cizgiKalinligi = 2;
	CvScalar cizgirengi = mavi;
	
	
	while (1){
		Ana_goruntu = 0;
		Ana_goruntu = cvQueryFrame(cap);
		cvFlip(Ana_goruntu, NULL, 1);
		if (imgKarala == NULL){
			imgKarala = cvCreateImage(cvGetSize(Ana_goruntu), 8, 3);
		}
		cvSmooth(Ana_goruntu, Ana_goruntu, CV_MEDIAN, 5, 5); //Arka Plan gürültüsünü azaltalým
		Mat AnaGoruntu2(Ana_goruntu);
		cv::Mat renkliGoruntu = cv::Mat::zeros(AnaGoruntu2.size(), AnaGoruntu2.type());

		//Mat nesnesinden IplImage nesnesine dönüþtürme iþlemi
		/*IplImage copy = renkliGoruntu;
		IplImage* newimage = &copy;*/
        pMOG2->operator()(AnaGoruntu2, fgMaskMOG2);
		AnaGoruntu2.copyTo(renkliGoruntu, fgMaskMOG2);
		deri = DeriRengi(renkliGoruntu);

		

		Mat sariTips, sariThresholded;
		cvtColor(deri, esikGoruntu, CV_BGR2HSV);
		cvtColor(esikGoruntu, esikGoruntu, CV_BGR2GRAY);
		equalizeHist(esikGoruntu, esikGoruntu);
		threshold(esikGoruntu, esikGoruntu, 250, 255, THRESH_BINARY + THRESH_OTSU);
		
		


		/* convexHUll geometri kýsmý,parmak uclarini algilama icin*/
		Mat kenarlar;
		int count = 0;
		char a[40];
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		//Canny(sariTips, kenarlar, 8, 8 * 2, 3);
		findContours(esikGoruntu, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat cizim = Mat::zeros(kenarlar.size(), CV_8UC3);
		if (contours.size() > 0){
			size_t indexOfBiggestContour = -1;
			size_t sizeOfBiggestContour = 0;
			for (size_t i = 0; i < contours.size(); i++)
			{
				if (contours[i].size() > sizeOfBiggestContour){
					sizeOfBiggestContour = contours[i].size();
					indexOfBiggestContour = i;
				}
			}
			vector<vector<int> >hull(contours.size());
			vector<vector<Point> >hullPoint(contours.size()); //elin hareketine göre eli çevreleyen çokgen	
			vector<vector<Vec4i> > defects(contours.size()); //parmak uclarindaki yesil noktalar..multi dimensional matrix
			vector<vector<Point> >defectPoint(contours.size()); //point olarak parmak ucu noktalarini x,y olarak tutuyor
			vector<vector<Point> >contours_poly(contours.size()); //eli çevreleyen hareketli dikdörtgen		
			Point2f rect_point[4];
			vector<RotatedRect>minRect(contours.size());
			vector<Rect> boundRect(contours.size());
			for (size_t i = 0; i < contours.size(); i++){
				if (contourArea(contours[i]) > 5000){
					convexHull(contours[i], hull[i], true);
					convexityDefects(contours[i], hull[i], defects[i]);
					if (indexOfBiggestContour == i){
						minRect[i] = minAreaRect(contours[i]);
						for (size_t k = 0; k < hull[i].size(); k++){
							int ind = hull[i][k];
							hullPoint[i].push_back(contours[i][ind]);
						}
						count = 0;

						for (size_t k = 0; k < defects[i].size() ; k++){ //defects[i].size()
							if (defects[i][k][3] > 13 * 256){
								int p_start = defects[i][k][0];
								int p_end = defects[i][k][1];
								int p_far = defects[i][k][2];
								defectPoint[i].push_back(contours[i][p_far]);
								
								count++;
								if (count == 1){
									circle(AnaGoruntu2, contours[i][count], 5, Scalar(0, 255, 255), 20);
								}//i ydi //circle(AnaGoruntu2, contours[i][p_end], 5, Scalar(0, 255, 255), 25); //i ydi
								else if (count == 2){
									count == 1;
									circle(AnaGoruntu2, contours[i][count], 5, Scalar(0, 255, 255), 20);
								}
									
								else if (count == 3)
								{
									count == 1;
									circle(AnaGoruntu2, contours[i][count], 5, Scalar(0, 255, 255), 20);
								}
								else if (count == 4)
								{
									count == 1;
									circle(AnaGoruntu2, contours[i][count], 5, Scalar(0, 255, 255), 20);
								}
								else if (count == 5 || count == 6)
								{
									count == 1;
									circle(AnaGoruntu2, contours[i][count], 5, Scalar(0, 255, 255), 20);
								}
								else
									strcpy_s(a, "EL GOSTER");
							}

						}
						

					}
				}
			}
		}





		/*convexHull bitis*/

		int sariLowH = 20;
		int sariLowS = 100;
		int sariLowV = 100;
		int sariHighH = 30;
		int sariHighS = 255;
		int sariHighV = 255;

		cvtColor(AnaGoruntu2, sariTips, CV_BGR2HSV);
		//cvtColor(sariTips, sariTips, CV_BGR2GRAY);
		inRange(sariTips, Scalar(sariLowH, sariLowS, sariLowV), Scalar(sariHighH, sariHighS, sariHighV), sariThresholded);
		//cvtColor(sariThresholded, sariThresholded, CV_BGR2GRAY);


		threshold(sariThresholded, sariThresholded, 10, 255, THRESH_BINARY + THRESH_OTSU);
		

		
		//IplImage kopya = esikGoruntu;
		//IplImage* imgEsik = cvCloneImage(&(IplImage)esikGoruntu);
		IplImage* sariFingerTipsEsik = cvCloneImage(&(IplImage)sariThresholded);
		

		CvMoments *anlikPos = (CvMoments*)malloc(sizeof(CvMoments));
		//cvMoments(imgEsik, anlikPos, 1);
		cvMoments(sariFingerTipsEsik, anlikPos, 1);
		
		//Gercek moment degerleri
		double moment10 = cvGetSpatialMoment(anlikPos, 1, 0);
		double moment01 = cvGetSpatialMoment(anlikPos, 0, 1);
		double alan = cvGetCentralMoment(anlikPos, 0, 0);

		//Geçmiþ ve anlýk pozisyonlarý tutma
		int sonX = pozisyonX;
		int sonY = pozisyonY;

		pozisyonX = 0;
		pozisyonY = 0;

		if (moment10 / alan >= 0 && moment10 / alan < 1280 && moment01 / alan >= 0 && moment01 / alan < 1280
			&& alan>alan_limit /* sýnýrý kontrol etmek*/)
		{
			pozisyonX = moment10 / alan;
			pozisyonY = moment01 / alan;
		}

		CvPoint yaziKonumu = cvPoint(150, 30);
		if (pozisyonX < 90 && pozisyonY > 400) // silgi-temizleme
		{
			cizgirengi = beyaz; // beyaz rengi silgi olarak kullanýyoruz
			cvPutText(Ana_goruntu, "Silgi Secildi.", yaziKonumu, &yazi, beyaz);
			sprintf(yazidizi, "Ekran Temizleniyor %d", temizle_sayac); // sayac asagi sayýyor ekran temizlenirken
			cvPutText(Ana_goruntu, yazidizi, cvPoint(150, 70), &yazi, kirmizi);
			temizle_sayac--;
			if (temizle_sayac < 0) // confirm in 10 frames before clearing
			{
				temizle_sayac = 20;
				sprintf(yazidizi, "d0%d.jpg", goruntu_kayit_no++);
				cvSaveImage(yazidizi,imgKarala); // frame'i goruntu olarak kaydet
				EkraniTemizle(imgKarala, imgSayfa);
				cvPutText(Ana_goruntu, "Ekran Temizlendi.", cvPoint(150, 110), &yazi, beyaz);
			}
		}
		else if (pozisyonX  > 540 && pozisyonY > 360)  // mavi renk secimi
		{
			cizgirengi = mavi;
			cvPutText(Ana_goruntu, "Mavi Renk Secildi.", yaziKonumu, &yazi, mavi);
		}

		else if (pozisyonX  > 540 && pozisyonY > 200 && pozisyonY< 280) // yeþil renk seçimi
		{
			cizgirengi = yesil;
			cvPutText(Ana_goruntu, "Yesil renk secildi.", yaziKonumu, &yazi, yesil);
		}

		else if (pozisyonX > 540 && pozisyonY < 120) // kirmizi renk secimi
		{
			cizgirengi = kirmizi;
			cvPutText(Ana_goruntu, "Kirmizi Renk Secildi.", yaziKonumu, &yazi, kirmizi);
		}

		else if (pozisyonX > 0 && pozisyonX  < 90 && pozisyonY > 0 && pozisyonY < 120) // çýkýþ
		{
			sprintf(yazidizi, "CIKIS %d", kapatma_sayac);
			cvPutText(Ana_goruntu, yazidizi, yaziKonumu, &yazi, kirmizi);
			kapatma_sayac--;
			if (kapatma_sayac < 0) // 10 frame'den sonra kapanýyor
				break;
		}
		else if (pozisyonX < 90 && pozisyonY > 130 && pozisyonY < 390) // çizgi kalýnlýðý ayarý
		{
			cizgiKalinligi = 6 - (pozisyonY / 60 - 1);  // change the thickness of line from 1 - 5 based on posY
		}

		sprintf(yazidizi, "%d", cizgiKalinligi);
		cvPutText(Ana_goruntu, yazidizi, cvPoint(40, 255), &buyukYazi, cizgirengi);

		double fark_X = sonX - pozisyonX;
		double fark_Y = sonY - pozisyonY;
		double magnitude = sqrt(pow(fark_X, 2) + pow(fark_Y, 2));
		// We want to draw a line only if its a valid position
		//if(lastX>0 && lastY>0 && posX>0 && posY>0)
		if (magnitude > 0 && magnitude < 100 && pozisyonX > 120 && pozisyonX<530)
		{
			
			cvLine(imgSayfa, cvPoint(pozisyonX, pozisyonY), cvPoint(sonX, sonY), cizgirengi, cizgiKalinligi, CV_AA);
		}

		// Add the scribbling image and the frame...
		cvAdd(imgSayfa, imgKarala,imgSayfa);

		// Combine everything in frame
		cvAnd(Ana_goruntu, imgSayfa, Ana_goruntu);
		cvAnd(imgRenkPaleti, Ana_goruntu, Ana_goruntu);

		cvShowImage("EsikDegeri", sariFingerTipsEsik);
		cvShowImage("Cizim", imgSayfa);
		cvShowImage("Video", Ana_goruntu);
		//imshow("ParmakTips", AnaGoruntu2);
		//imshow("SariThresh", sariThresholded);


		int c = cvWaitKey(10);
		if (c == 27)  //ESC key
			break;
		//else if(c==49) // 1 key


		cvReleaseImage(&sariFingerTipsEsik);
		delete anlikPos;


		//Sonuclarý goster
	    
		//imshow("orjinal", AnaGoruntu2);
		//imshow("Threshold", esikGoruntu);
		//imshow("Deri algilayici", deri);
		//cvShowImage("video",Ana_goruntu);

		//char key = (char)waitKey(100);
		//if (key == 27)
			//break;

	}

	
	cvReleaseCapture(&cap);
	cvReleaseImage(&imgRenkPaleti);
	cvReleaseImage(&imgKarala);
	return 0;
}