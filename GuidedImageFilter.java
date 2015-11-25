import org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.*;

public class GuidedImageFilter {
	
   public static IplImage guidedfilter_color(IplImage I, IplImage p, int r, double eps){
	   int wid = I.width();
	   int hei = I.height();
	   
	   // normalization
	   IplImage ones = cvCreateImage(cvSize(wid,hei), IPL_DEPTH_32F, 1);
	   cvZero(ones);
	   cvAddS(ones, cvScalar(1.0), ones, null);
	   IplImage N = boxfilter(ones, r);
	   
	   IplImage bImg32 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage gImg32 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage rImg32 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvSplit(I, bImg32, gImg32, rImg32, null);	   	  
	   
	   IplImage mean_I_b = boxfilter(bImg32, r);
	   cvDiv(mean_I_b, N, mean_I_b, 1);	   	  
	   
	   IplImage mean_I_g = boxfilter(gImg32, r);
	   cvDiv(mean_I_g, N, mean_I_g, 1);
	   
	   IplImage mean_I_r = boxfilter(rImg32, r);
	   cvDiv(mean_I_r, N, mean_I_r, 1);
	   
	   IplImage mean_p = boxfilter(p, r);
	   cvDiv(mean_p, N, mean_p, 1);
       
	   IplImage mean_Ip_b = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(bImg32, p, mean_Ip_b, 1);
	   mean_Ip_b = boxfilter(mean_Ip_b, r);
	   cvDiv(mean_Ip_b, N, mean_Ip_b, 1);
	   
	   IplImage mean_Ip_g = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(gImg32, p, mean_Ip_g, 1);
	   mean_Ip_g = boxfilter(mean_Ip_g, r);
	   cvDiv(mean_Ip_g, N, mean_Ip_g, 1);
	   
	   IplImage mean_Ip_r = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(rImg32, p, mean_Ip_r, 1);
	   mean_Ip_r = boxfilter(mean_Ip_r, r);
	   cvDiv(mean_Ip_r, N, mean_Ip_r, 1);	   	  
	   
	   IplImage cov_Ip_b = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(mean_I_b, mean_p, cov_Ip_b, 1);
	   cvSub(mean_Ip_b, cov_Ip_b, cov_Ip_b, null);
	   
	   IplImage cov_Ip_g = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(mean_I_g, mean_p, cov_Ip_g, 1);
	   cvSub(mean_Ip_g, cov_Ip_g, cov_Ip_g, null);
	   
	   IplImage cov_Ip_r = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(mean_I_r, mean_p, cov_Ip_r, 1);
	   cvSub(mean_Ip_r, cov_Ip_r, cov_Ip_r, null);
	   
	   /*
        * variance of I in each local patch: the matrix Sigma in Eqn (14).
        * Note the variance in each local patch is a 3x3 symmetric matrix:
        *        rr, rg, rb
        * Sigma = rg, gg, gb
        *        rb, gb, bb
        */
	   
	   IplImage mean_I_r_MUL_mean_I_r = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage mean_I_r_MUL_mean_I_g = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage mean_I_r_MUL_mean_I_b = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage mean_I_g_MUL_mean_I_g = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage mean_I_g_MUL_mean_I_b = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage mean_I_b_MUL_mean_I_b = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   
	   IplImage var_I_rr = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(rImg32, rImg32, var_I_rr, 1);
	   var_I_rr = boxfilter(var_I_rr, r);
	   cvDiv(var_I_rr, N, var_I_rr, 1);
	   cvMul(mean_I_r, mean_I_r, mean_I_r_MUL_mean_I_r, 1);
	   cvSub(var_I_rr, mean_I_r_MUL_mean_I_r, var_I_rr, null);
	   
	   IplImage var_I_rg = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(rImg32, gImg32, var_I_rg, 1);
	   var_I_rg = boxfilter(var_I_rg, r);
	   cvDiv(var_I_rg, N, var_I_rg, 1);
	   cvMul(mean_I_r, mean_I_g, mean_I_r_MUL_mean_I_g, 1);
	   cvSub(var_I_rg, mean_I_r_MUL_mean_I_g, var_I_rg, null);	   	   
	   
	   IplImage var_I_rb = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(rImg32, bImg32, var_I_rb, 1);
	   var_I_rb = boxfilter(var_I_rb, r);
	   cvDiv(var_I_rb, N, var_I_rb, 1);
	   cvMul(mean_I_r, mean_I_b, mean_I_r_MUL_mean_I_b, 1);
	   cvSub(var_I_rb, mean_I_r_MUL_mean_I_b, var_I_rb, null);
	   
	   IplImage var_I_gg = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(gImg32, gImg32, var_I_gg, 1);
	   var_I_gg = boxfilter(var_I_gg, r);
	   cvDiv(var_I_gg, N, var_I_gg, 1);
	   cvMul(mean_I_g, mean_I_g, mean_I_g_MUL_mean_I_g, 1);
	   cvSub(var_I_gg, mean_I_g_MUL_mean_I_g, var_I_gg, null);
	   
	   IplImage var_I_gb = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(gImg32, bImg32, var_I_gb, 1);
	   var_I_gb = boxfilter(var_I_gb, r);
	   cvDiv(var_I_gb, N, var_I_gb, 1);
	   cvMul(mean_I_g, mean_I_b, mean_I_g_MUL_mean_I_b, 1);
	   cvSub(var_I_gb, mean_I_g_MUL_mean_I_b, var_I_gb, null);	  	   
	   
	   IplImage var_I_bb = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(bImg32, bImg32, var_I_bb, 1);
	   var_I_bb = boxfilter(var_I_bb, r);
	   cvDiv(var_I_bb, N, var_I_bb, 1);
	   cvMul(mean_I_b, mean_I_b, mean_I_b_MUL_mean_I_b, 1);
	   cvSub(var_I_bb, mean_I_b_MUL_mean_I_b, var_I_bb, null);
	   
	   IplImage a = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 3);
	   IplImage eye = cvCreateImage(cvSize(3, 3), IPL_DEPTH_32F, 1);
	   cvZero(eye);
	   cvSet2D(eye, 0, 0, cvScalar(1));
	   cvSet2D(eye, 1, 1, cvScalar(1));
	   cvSet2D(eye, 2, 2, cvScalar(1));
	   cvMul(eye, eye, eye, eps);
	   for(int y=0; y<hei; y++){
		   for(int x=0; x<wid; x++){
			   IplImage Sigma = cvCreateImage(cvSize(3, 3), IPL_DEPTH_32F, 1);			   
			   cvSet2D(Sigma, 0, 0, cvGet2D(var_I_rr, y, x));
			   cvSet2D(Sigma, 0, 1, cvGet2D(var_I_rg, y, x));
			   cvSet2D(Sigma, 0, 2, cvGet2D(var_I_rb, y, x));
			   cvSet2D(Sigma, 1, 0, cvGet2D(var_I_rg, y, x));
			   cvSet2D(Sigma, 1, 1, cvGet2D(var_I_gg, y, x));
			   cvSet2D(Sigma, 1, 2, cvGet2D(var_I_gb, y, x));
			   cvSet2D(Sigma, 2, 0, cvGet2D(var_I_rb, y, x));
			   cvSet2D(Sigma, 2, 1, cvGet2D(var_I_gb, y, x));
			   cvSet2D(Sigma, 2, 2, cvGet2D(var_I_bb, y, x));
			   
			   IplImage cov_Ip = cvCreateImage(cvSize(3, 1), IPL_DEPTH_32F, 1);
			   cvSet2D(cov_Ip, 0, 0, cvGet2D(cov_Ip_r, y, x));
			   cvSet2D(cov_Ip, 0, 1, cvGet2D(cov_Ip_g, y, x));
			   cvSet2D(cov_Ip, 0, 2, cvGet2D(cov_Ip_b, y, x));
			   
			   cvAdd(Sigma, eye, Sigma, null);
			   cvInvert(Sigma, Sigma, CV_LU);
			   IplImage temp = cvCreateImage(cvSize(3, 1), IPL_DEPTH_32F, 1);
			   cvGEMM(cov_Ip, Sigma, 1, null, 0, temp);
			   CvScalar scalar = new CvScalar();
			   scalar.setVal(0, cvGet2D(temp, 0, 0).val(0));
			   scalar.setVal(1, cvGet2D(temp, 0, 1).val(0));
			   scalar.setVal(2, cvGet2D(temp, 0, 2).val(0));
			   cvSet2D(a, y, x, scalar);
			   
			   cvReleaseImage(Sigma);
			   cvReleaseImage(cov_Ip);
			   cvReleaseImage(temp);
		   }
	   }
	   
	   IplImage b = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage ak1 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage ak2 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage ak3 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage temp1 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage temp2 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage temp3 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvSplit(a, ak1, ak2, ak3, null);
	   cvMul(ak1, mean_I_r, temp1, 1);
	   cvMul(ak2, mean_I_g, temp2, 1);
	   cvMul(ak3, mean_I_b, temp3, 1);
	   cvSub(mean_p, temp1, b, null);
	   cvSub(b, temp2, b, null);
	   cvSub(b, temp3, b, null);
	   
	   IplImage q = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage term1 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage term2 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage term3 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   IplImage term4 = cvCreateImage(cvSize(wid, hei), IPL_DEPTH_32F, 1);
	   cvMul(boxfilter(ak1, r), rImg32, term1, 1);
	   cvMul(boxfilter(ak2, r), gImg32, term2, 1);
	   cvMul(boxfilter(ak3, r), bImg32, term3, 1);
	   term4 = boxfilter(b, r);	   
	   cvAdd(term1, term2, q, null);
	   cvAdd(q, term3, q, null);
	   cvAdd(q, term4, q, null);
	   cvDiv(q, N, q, 1);
	   
	   // release the images
	   cvReleaseImage(ones);
	   cvReleaseImage(N);
	   cvReleaseImage(bImg32);
	   cvReleaseImage(gImg32);
	   cvReleaseImage(rImg32);
	   cvReleaseImage(mean_I_b);
	   cvReleaseImage(mean_I_g);
	   cvReleaseImage(mean_I_r);
	   cvReleaseImage(mean_p);
	   cvReleaseImage(mean_Ip_b);
	   cvReleaseImage(mean_Ip_g);
	   cvReleaseImage(mean_Ip_r);
	   cvReleaseImage(cov_Ip_b);
	   cvReleaseImage(cov_Ip_g);
	   cvReleaseImage(cov_Ip_r);
	   cvReleaseImage(mean_I_r_MUL_mean_I_g);
	   cvReleaseImage(mean_I_r_MUL_mean_I_b);
	   cvReleaseImage(mean_I_g_MUL_mean_I_g);
	   cvReleaseImage(mean_I_g_MUL_mean_I_b);
	   cvReleaseImage(mean_I_b_MUL_mean_I_b);
	   cvReleaseImage(var_I_rr);
	   cvReleaseImage(var_I_rg);
	   cvReleaseImage(var_I_rb);
	   cvReleaseImage(var_I_gg);
	   cvReleaseImage(var_I_gb);
	   cvReleaseImage(var_I_bb);
	   cvReleaseImage(a);	   
	   cvReleaseImage(eye);
	   cvReleaseImage(b);
	   cvReleaseImage(ak1);
	   cvReleaseImage(ak2);
	   cvReleaseImage(ak3);
	   cvReleaseImage(temp1);
	   cvReleaseImage(temp2);
	   cvReleaseImage(temp3);
	   cvReleaseImage(term1);
	   cvReleaseImage(term2);
	   cvReleaseImage(term3);
	   cvReleaseImage(term4);
	   return q;
   }  
	
	public static IplImage boxfilter(IplImage imgSrc, int r){
		CvSize size = cvGetSize(imgSrc);
		IplImage imgDst = cvCreateImage(size, IPL_DEPTH_32F, 1);
		cvZero(imgDst);
		IplImage imgCum1 = cvCreateImage(size, IPL_DEPTH_32F, 1);
		IplImage imgCum2 = cvCreateImage(size, IPL_DEPTH_32F, 1);
		// cumulative sum over Y axis
		imgCum1 = cumsum(imgSrc, 1);
		//UtilClass.printImg2(imgCum1); 
		cvCopy(imgCum1, imgCum2, null);
		// difference over Y axis		
		cvSetImageROI(imgDst, cvRect(0, 0, imgCum1.width(), r+1));
		cvSetImageROI(imgCum1, cvRect(0, r+1-1, imgCum1.width(), r+1));
		cvCopy(imgCum1, imgDst, null);
		cvResetImageROI(imgDst);
		cvResetImageROI(imgCum1);		
		
		// when the number of rows is larger than zero
		if(imgCum1.height() - r - r - 2 + 1 > 0){
			cvSetImageROI(imgDst, cvRect(0, r+2-1,imgCum1.width(), imgCum1.height()-r-r-2+1));
			cvSetImageROI(imgCum1, cvRect(0, 2*r+2-1,imgCum1.width(), imgCum1.height()-r-r-2+1));
			cvSetImageROI(imgCum2, cvRect(0, 1-1,imgCum1.width(), imgCum1.height()-r-r-2+1));
			cvSub(imgCum1, imgCum2, imgDst, null);
			cvResetImageROI(imgDst);
			cvResetImageROI(imgCum1);
			cvResetImageROI(imgCum2);
		}
		
		cvSetImageROI(imgDst, cvRect(0, imgCum1.height()-r+1-1, imgCum1.width(), r-1+1));		
		IplImage repmat = cvCreateImage(cvSize(imgCum1.width(), r), IPL_DEPTH_32F, 1);		
		for(int i=0; i < r; i++){
			for(int j=0; j<imgCum1.width(); j++){
				double value = cvGet2D(imgCum1, imgCum1.height()-1, j).val(0);
				CvScalar scalar = new CvScalar();
				scalar.setVal(0, value);
				cvSet2D(repmat, i, j, scalar);
			}
		}		
		cvSetImageROI(imgCum2, cvRect(0, imgCum2.height()-2*r-1, imgCum1.width(), r-1+1));
		cvSub(repmat, imgCum2, imgDst);
		cvResetImageROI(imgDst);
		cvResetImageROI(imgCum1);
		cvResetImageROI(imgCum2);
		//UtilClass.printImg2(imgDst);
		
		// cumulative sum over X axis
		imgCum1 = cumsum(imgDst, 2);
		cvCopy(imgCum1, imgCum2, null);
		// difference over X axis
		cvSetImageROI(imgDst, cvRect(0, 0, r+1, imgCum1.height()));
		cvSetImageROI(imgCum1, cvRect(r+1-1, 0, r+1, imgCum1.height()));
		cvCopy(imgCum1, imgDst, null);
		cvResetImageROI(imgDst);
		cvResetImageROI(imgCum1);
		
		// when the number of rows is larger than zero
		if(imgCum1.width()-r-r-2+1>0){
			cvSetImageROI(imgDst, cvRect(r+2-1, 0, imgCum1.width()-r-r-2+1, imgCum1.height()));
			cvSetImageROI(imgCum1, cvRect(2*r+2-1, 0,  imgCum1.width()-r-r-2+1, imgCum1.height()));
			cvSetImageROI(imgCum2, cvRect(1-1, 0, imgCum2.width()-r-r-2+1, imgCum1.height()));
			cvSub(imgCum1, imgCum2, imgDst, null);
			cvResetImageROI(imgDst);
			cvResetImageROI(imgCum1);
			cvResetImageROI(imgCum2);
		}
		cvSetImageROI(imgDst, cvRect(imgCum1.width()-r+1-1, 0, r-1+1, imgCum1.height()));		
		repmat = cvCreateImage(cvSize(r, imgCum1.height()), IPL_DEPTH_32F, 1);
		for(int i = 0; i<imgCum1.height(); i++){
			for(int j = 0; j < r; j++){
				double value = cvGet2D(imgCum1, i, imgCum1.width()-1).val(0);
				CvScalar scalar = new CvScalar();
				scalar.setVal(0, value);
				cvSet2D(repmat, i, j, scalar);
			}
		}
		cvSetImageROI(imgCum2, cvRect(imgCum2.width()-2*r-1, 0, r-1+1, imgCum1.height()));
		cvSub(repmat, imgCum2, imgDst, null);
		cvResetImageROI(imgDst);
		cvResetImageROI(imgCum1);
		cvResetImageROI(imgCum2);
		
		cvReleaseImage(imgCum1);
		cvReleaseImage(imgCum2);
		cvReleaseImage(repmat);
    	return imgDst;
    }
	
	public static IplImage cumsum(IplImage imgSrc, int dimension){
		IplImage imgDst = cvCreateImage(cvSize(imgSrc.width(), imgSrc.height()), IPL_DEPTH_32F, 1);
		cvZero(imgDst);
		if(dimension == 1){
			for(int y = 0; y < imgSrc.height(); y++){
				for(int x = 0; x < imgSrc.width(); x++){
					if(y == 0){						
						double value = cvGet2D(imgSrc, y, x).val(0);
						CvScalar scalar = new CvScalar();
						scalar.setVal(0, value);
						cvSet2D(imgDst, y, x, scalar);
					}else{
						double value = cvGet2D(imgSrc, y, x).val(0) + 
								cvGet2D(imgDst, y-1, x).val(0);
						//System.out.println("("+y+", "+x+"): "+cvGet2D(imgSrc, y, x).val(0)+", "+cvGet2D(imgDst, y-1, x).val(0)+" = " + value);						
						CvScalar scalar = new CvScalar();
						scalar.setVal(0, value);
						cvSet2D(imgDst, y, x, scalar);
					}
				} 
			}			
		}else{
			for(int y = 0; y < imgSrc.height(); y++){
				for(int x = 0; x < imgSrc.width(); x++){
					if(x == 0){
						double value = cvGet2D(imgSrc, y, x).val(0);
						CvScalar scalar = new CvScalar();
						scalar.setVal(0, value);
						cvSet2D(imgDst, y, x, scalar);
					}else{
						double value = cvGet2D(imgSrc, y, x).val(0) + 
								cvGet2D(imgDst, y, x-1).val(0);
						//System.out.println("("+y+", "+x+"): "+cvGet2D(imgSrc, y, x).val(0)+", "+cvGet2D(imgDst, y, x-1).val(0)+" = " + value);	
						CvScalar scalar = new CvScalar();
						scalar.setVal(0, value);
						cvSet2D(imgDst, y, x, scalar);
					}
				}
			}
		}
		return imgDst;
	}
	
	public static void imgBgrDoubleNormalize(IplImage src, IplImage des){
		for(int i = 0; i < des.height(); i++){
			for(int j = 0; j < des.width(); j++){
				CvScalar rgb = cvGet2D(src, i, j);
				double b = rgb.val(0)/255.0;
				double g = rgb.val(1)/255.0;
				double r = rgb.val(2)/255.0;
				CvScalar scalar = new CvScalar();
				scalar.setVal(0, b);
				scalar.setVal(1, g);
				scalar.setVal(2, r);
				cvSet2D(des, i, j, scalar);
			}
		}
	}
}
