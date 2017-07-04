/*********************************************************************
 * trackFeatures.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>		/* fabs() */
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */

/* Our includes */
#include "base.h"
#include "klt.h"

extern int KLT_verbose;

typedef float *_FloatWindow;

/*********************************************************************
 * _interpolate
 * 
 * Given a point (x,y) in an image, computes the bilinear interpolated 
 * gray-level value of the point in the image.  
 */

static float _interpolate(
  float x, 
  float y, 
  FLOAT32* img,
  int ncols,
  int nrows)
{
  int xt = (int) x;  /* coordinates of top-left corner */
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;
  float *ptr = img + (ncols*yt) + xt;

#ifndef _DNDEBUG
//  if (xt<0 || yt<0 || xt>=img->ncols-1 || yt>=img->nrows-1) {
//    printf( "(xt,yt)=(%d,%d)  imgsize=(%d,%d)\n"
//            "(x,y)=(%f,%f)  (ax,ay)=(%f,%f)\n",
//            xt, yt, img->ncols, img->nrows, x, y, ax, ay);
//  }
#endif

//  assert (xt >= 0 && yt >= 0 && xt <= img->ncols - 2 && yt <= img->nrows - 2);

  return ( (1-ax) * (1-ay) * *ptr +
           ax   * (1-ay) * *(ptr+1) +
           (1-ax) *   ay   * *(ptr+ncols) +
           ax   *   ay   * *(ptr+ncols+1) );
}


/*********************************************************************
 * _computeIntensityDifference
 *
 * Given two images and the window center in both images,
 * aligns the images wrt the window and computes the difference 
 * between the two overlaid images.
 */

static void _computeIntensityDifference(
  FLOAT32* img1,   /* images */
  FLOAT32* img2,
  int ncols, int nrows,
  float x1, float y1,     /* center of window in 1st img */
  float x2, float y2,     /* center of window in 2nd img */
  int width, int height,  /* size of window */
  _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1, ncols, nrows);
      g2 = _interpolate(x2+i, y2+j, img2, ncols, nrows);
      *imgdiff++ = g1 - g2;
    }
}


/*********************************************************************
 * _computeGradientSum
 *
 * Given two gradients and the window center in both images,
 * aligns the gradients wrt the window and computes the sum of the two 
 * overlaid gradients.
 */

static void _computeGradientSum(
  FLOAT32* gradx1,  /* gradient images */
  FLOAT32* grady1,
  FLOAT32* gradx2,
  FLOAT32* grady2,
  int ncols, int nrows,
  float x1, float y1,      /* center of window in 1st img */
  float x2, float y2,      /* center of window in 2nd img */
  int width, int height,   /* size of window */
  _FloatWindow gradx,      /* output */
  _FloatWindow grady)      /*   " */
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, gradx1, ncols, nrows);
      g2 = _interpolate(x2+i, y2+j, gradx2, ncols, nrows);
      *gradx++ = g1 + g2;
      g1 = _interpolate(x1+i, y1+j, grady1, ncols, nrows);
      g2 = _interpolate(x2+i, y2+j, grady2, ncols, nrows);
      *grady++ = g1 + g2;
    }
}





/*********************************************************************
 * _compute2by2GradientMatrix
 *
 */

static void _compute2by2GradientMatrix(
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float *gxx,  /* return values */
  float *gxy, 
  float *gyy) 

{
  register float gx, gy;
  register int i;

  /* Compute values */
  *gxx = 0.0;  *gxy = 0.0;  *gyy = 0.0;
  for (i = 0 ; i < width * height ; i++)  {
    gx = *gradx++;
    gy = *grady++;
    *gxx += gx*gx;
    *gxy += gx*gy;
    *gyy += gy*gy;
  }
}
	


//--------------------------------------------------------------------


static void _computeIntensityInterpolate_ref(
                                        FLOAT32* img1,   /* images */
                                        int ncols, int nrows,
                                        float x1, float y1,     /* center of window in 1st img */
                                        int width, int height,  /* size of window */
                                        float*I)   /* output */
{
    register int hw = width/2, hh = height/2;
    register int i, j;
    
    /* Compute values */
    for (j = -hh ; j <= hh ; j++){
        for (i = -hw ; i <= hw ; i++)  {
            *I++ = _interpolate(x1+i, y1+j, img1, ncols, nrows);
        }
    }
}


static void _computeIntensityInterpolate_diff(
                                        FLOAT32* I,   /* images */
                                        FLOAT32* img2,
                                        int ncols, int nrows,
                                        float x2, float y2,     /* center of window in 2nd img */
                                        int width, int height,  /* size of window */
                                        _FloatWindow imgdiff)   /* output */
{
    register int hw = width/2, hh = height/2;
    register int i, j;
    
    /* Compute values */
    for (j = -hh ; j <= hh ; j++){
        for (i = -hw ; i <= hw ; i++)  {
            *imgdiff++ = *I++ - _interpolate(x2+i, y2+j, img2, ncols, nrows);
        }
    }
}



static void _computeGradientSum_ref(
                                FLOAT32* gradx1,  //
                                FLOAT32* grady1,
                                int ncols, int nrows,
                                float x1, float y1,      //
                                int width, int height,   //
                                _FloatWindow gradx,      //
                                _FloatWindow grady,
                                 float *gxx,  /* return values */
                                 float *gxy,
                                 float *gyy
                                )
{
    register int hw = width/2, hh = height/2;
    float g1, g2;
    register int i, j;
    
    /* Compute values */
    for (j = -hh ; j <= hh ; j++){
        for (i = -hw ; i <= hw ; i++)  {
            g1 = _interpolate(x1+i, y1+j, gradx1, ncols, nrows);
            g2 = _interpolate(x1+i, y1+j, grady1, ncols, nrows);
            *gradx++=g1;
            *grady++=g2;
            * gxx+=g1*g1;
            * gxy+=g1*g2;
            * gyy+=g2*g2;
            
        }
    }
}

static int _solveEquation_(
                          float gxx, float gxy, float gyy,
                          float ex, float ey,
                          float small,
                          float *dx, float *dy, float det_inv)
{
    
    *dx = (gyy*ex - gxy*ey)*det_inv;
    *dy = (gxx*ey - gxy*ex)*det_inv;
    return KLT_TRACKED;
}


//--------------------------------------------------------------------


/*********************************************************************
 * _compute2by1ErrorVector
 *
 */

static void _compute2by1ErrorVector(
  _FloatWindow imgdiff,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
  float *ex,   /* return values */
  float *ey)
{
  register float diff;
  register int i;

  /* Compute values */
  *ex = 0;  *ey = 0;  
  for (i = 0 ; i < width * height ; i++)  {
    diff = *imgdiff++;
    *ex += diff * (*gradx++);
    *ey += diff * (*grady++);
  }
  *ex *= step_factor;
  *ey *= step_factor;
}


/*********************************************************************
 * _solveEquation
 *
 * Solves the 2x2 matrix equation
 *         [gxx gxy] [dx] = [ex]
 *         [gxy gyy] [dy] = [ey]
 * for dx and dy.
 *
 * Returns KLT_TRACKED on success and KLT_SMALL_DET on failure
 */

static int _solveEquation(
  float gxx, float gxy, float gyy,
  float ex, float ey,
  float small,
  float *dx, float *dy)
{
  float det = gxx*gyy - gxy*gxy;
	
  if (det < small)  return KLT_SMALL_DET;

  *dx = (gyy*ex - gxy*ey)/det;
  *dy = (gxx*ey - gxy*ex)/det;
  return KLT_TRACKED;
}


/*********************************************************************
 * _allocateFloatWindow
 */
	
static _FloatWindow _allocateFloatWindow(
  int width,
  int height)
{
  _FloatWindow fw;

  fw = (_FloatWindow) malloc(width*height*sizeof(float));
  return fw;
}



/*********************************************************************
 * _sumAbsFloatWindow
 */

static float _sumAbsFloatWindow(
  _FloatWindow fw,
  int width,
  int height)
{
  float sum = 0.0;
  int w;

  for ( ; height > 0 ; height--)
    for (w=0 ; w < width ; w++)
      sum += (float) fabs(*fw++);

  return sum;
}


/*********************************************************************
 * _trackFeature
 *
 * Tracks a feature point from one image to the next.
 *
 * RETURNS
 * KLT_SMALL_DET if feature is lost,
 * KLT_MAX_ITERATIONS if tracking stopped because iterations timed out,
 * KLT_TRACKED otherwise.
 */

static int _trackFeature(
  float x1,  /* location of window in first image */
  float y1,
  float *x2, /* starting location of search in second image */
  float *y2,
  FLOAT32* img1, 
  FLOAT32* gradx1,
  FLOAT32* grady1,
  FLOAT32* img2, 
  FLOAT32* gradx2,
  FLOAT32* grady2,
  int ncols, int nrows,
  int width,           /* size of window */
  int height,
  float step_factor, /* 2.0 comes from equations, 1.0 seems to avoid overshooting */
  int max_iterations,
  float small,         /* determinant threshold for declaring KLT_SMALL_DET */
  float th,            /* displacement threshold for stopping               */
  float max_residue,   /* residue threshold for declaring KLT_LARGE_RESIDUE */
  int lighting_insensitive)  /* whether to normalize for gain and bias */
{
    
  float *I, *dxx, *dxy, *dyy ;
  _FloatWindow imgdiff, gradx, grady;
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status;
  int hw = width/2;
  int hh = height/2;
  int nc = ncols;
  int nr = nrows;
  float one_plus_eps = 1.001f;   /* To prevent rounding errors */
  float det=1;
  float det_inv=1;
	
  /* Allocate memory for windows */
  imgdiff = _allocateFloatWindow(width, height);
  gradx   = _allocateFloatWindow(width, height);
  grady   = _allocateFloatWindow(width, height);

  I = calloc(width*height ,sizeof(float));
  dxx = calloc(width*height ,sizeof(float));
  dxy = calloc(width*height ,sizeof(float));
  dyy = calloc(width*height ,sizeof(float));

    
 // initial Process
 _computeIntensityInterpolate_ref(img1,  ncols, nrows, x1, y1,
                                width, height, I);
 _computeGradientSum_ref(gradx1, grady1, ncols, nrows, x1, y1,
                                width, height, gradx, grady, &gxx, &gxy, &gyy);

  gxx/=(width*height);
  gxy/=(width*height);
  gyy/=(width*height);
    
  det = gxx*gyy - gxy*gxy;
  if (det < small)  return KLT_SMALL_DET;
  
  det_inv = 1.0/det;
    
  /* Iteratively update the window position */
  do  {

    /* If out of bounds, exit loop */
    if (  x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
         *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
          y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
         *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
      status = KLT_OOB;
      break;
    }

      _computeIntensityInterpolate_diff(I,img2,ncols, nrows, *x2, *y2,width, height, imgdiff);
      
//      _computeIntensityDifference(img1, img2, ncols, nrows, x1, y1, *x2, *y2,
//                                  width, height, imgdiff);
//      _computeGradientSum(gradx1, grady1, gradx2, grady2, ncols, nrows,
//			  x1, y1, *x2, *y2, width, height, gradx, grady);
		

    /* Use these windows to construct matrices */
//    _compute2by2GradientMatrix(gradx, grady, width, height,
//                               &gxx, &gxy, &gyy);
    _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
                            &ex, &ey);
				
    /* Using matrices, solve equation for new displacement */
    //status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
      status = _solveEquation_(gxx, gxy, gyy, ex, ey, small, &dx, &dy, det_inv);
      
    //if (status == KLT_SMALL_DET)  break;

    *x2 += dx;
    *y2 += dy;
    iteration++;

  }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);

  /* Check whether window is out of bounds */
  if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps || 
      *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
    status = KLT_OOB;

  /* Check whether residue is too large */
  if (status == KLT_TRACKED)  {
      _computeIntensityDifference(img1, img2, ncols, nrows, x1, y1, *x2, *y2,
                                  width, height, imgdiff);
    if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue) 
      status = KLT_LARGE_RESIDUE;
  }

  /* Free memory */
  free(imgdiff);  free(gradx);  free(grady);

  /* Return appropriate value */
  if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
  else if (status == KLT_OOB)  return KLT_OOB;
  else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
  else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
  else  return KLT_TRACKED;

}

/*********************************************************************
 * KLTTrackFeatures
 *
 * Tracks feature points from one image to the next.
 */

void KLTTrackFeatures(
					  KLT_TrackingContext tc,
					  UINT8 *img1,
					  UINT8 *img2,
					  int ncols,
					  int nrows,
					  KLT_FeatureList featurelist)
{
	FLOAT32* tmpimg, *floatimg1, *floatimg2;
	FLOAT32** pyramid1, **pyramid1_gradx, **pyramid1_grady,
		**pyramid2, **pyramid2_gradx, **pyramid2_grady;
	float subsampling = (float) tc->subsampling;
	float xloc, yloc, xlocout, ylocout;
	int val;
	int indx, r;
	KLT_BOOL floatimg1_created = FALSE;
	int i;


	if (tc->window_width % 2 != 1)  tc->window_width = tc->window_width+1;
	if (tc->window_height % 2 != 1) tc->window_height = tc->window_height+1;
	if (tc->window_width < 3)       tc->window_width = 3;
	if (tc->window_height < 3)      tc->window_height = 3;
	

	/* Create temporary image */
	tmpimg = _KLTCreateFloatImage(ncols, nrows);

	/* Process first image by converting to float, smoothing, computing */
	/* pyramid, and computing gradient pyramids */
	if (tc->sequentialMode && tc->pyramid_last != NULL)  {
		pyramid1 = (FLOAT32**) tc->pyramid_last;
		pyramid1_gradx = (FLOAT32**) tc->pyramid_last_gradx;
		pyramid1_grady = (FLOAT32**) tc->pyramid_last_grady;
		assert(pyramid1_gradx != NULL);
		assert(pyramid1_grady != NULL);
	} else  {
		floatimg1_created = TRUE;
		floatimg1 = _KLTCreateFloatImage(ncols, nrows);
		_KLTToFloatImage(img1, ncols, nrows, tmpimg);
		_KLTComputeSmoothedImage(tmpimg, ncols, nrows, _KLTComputeSmoothSigma(tc), floatimg1);
		pyramid1 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
		_KLTComputePyramid(floatimg1, ncols, nrows, (int) subsampling, tc->nPyramidLevels, pyramid1, tc->pyramid_sigma_fact);
		pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
		pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
		for (i = 0 ; i < tc->nPyramidLevels ; i++)
			_KLTComputeGradients(pyramid1[i], ncols, nrows,  tc->grad_sigma,
			pyramid1_gradx[i],
			pyramid1_grady[i]);
	}

	/* Do the same thing with second image */
	floatimg2 = _KLTCreateFloatImage(ncols, nrows);
	_KLTToFloatImage(img2, ncols, nrows, tmpimg);
	_KLTComputeSmoothedImage(tmpimg, ncols, nrows, _KLTComputeSmoothSigma(tc), floatimg2);
	pyramid2 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
	_KLTComputePyramid(floatimg2, ncols, nrows, (int) subsampling, tc->nPyramidLevels, pyramid2, tc->pyramid_sigma_fact);
	pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
	pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
	for (i = 0 ; i < tc->nPyramidLevels ; i++)
		_KLTComputeGradients(pyramid2[i], ncols, nrows, tc->grad_sigma,
		pyramid2_gradx[i],
		pyramid2_grady[i]);
    
	/* For each feature, do ... */
	for (indx = 0 ; indx < featurelist->nFeatures ; indx++)  {

		/* Only track features that are not lost */
		if (featurelist->feature[indx]->val >= 0)  {

			xloc = featurelist->feature[indx]->x;
			yloc = featurelist->feature[indx]->y;

			/* Transform location to coarsest resolution */
			for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
				xloc /= subsampling;  yloc /= subsampling;
			}
			xlocout = xloc;  ylocout = yloc;

			/* Beginning with coarsest resolution, do ... */
			for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {

				/* Track feature at current resolution */
				xloc *= subsampling;  yloc *= subsampling;
				xlocout *= subsampling;  ylocout *= subsampling;

				val = _trackFeature(xloc, yloc, 
					&xlocout, &ylocout,
					pyramid1[r],
					pyramid1_gradx[r], pyramid1_grady[r],
					pyramid2[r],
					pyramid2_gradx[r], pyramid2_grady[r],
                    ncols, nrows,
					tc->window_width, tc->window_height,
					tc->step_factor,
					tc->max_iterations,
					tc->min_determinant,
					tc->min_displacement,
					tc->max_residue,
					tc->lighting_insensitive);

				if (val==KLT_SMALL_DET || val==KLT_OOB)
					break;

			}
		}
	}

	if (tc->sequentialMode)  {
		tc->pyramid_last = pyramid2;
		tc->pyramid_last_gradx = pyramid2_gradx;
		tc->pyramid_last_grady = pyramid2_grady;
	} else  {
		_KLTFreePyramid(pyramid2, tc->nPyramidLevels);
		_KLTFreePyramid(pyramid2_gradx, tc->nPyramidLevels);
		_KLTFreePyramid(pyramid2_grady, tc->nPyramidLevels);
	}

	/* Free memory */
	_KLTFreeFloatImage(tmpimg);
	if (floatimg1_created)  _KLTFreeFloatImage(floatimg1);
	_KLTFreeFloatImage(floatimg2);
	_KLTFreePyramid(pyramid1, tc->nPyramidLevels);
	_KLTFreePyramid(pyramid1_gradx, tc->nPyramidLevels);
	_KLTFreePyramid(pyramid1_grady, tc->nPyramidLevels);

	if (KLT_verbose >= 1)  {
		printf(  "\n\t%d features successfully tracked.\n",
			KLTCountRemainingFeatures(featurelist));
		if (tc->writeInternalImages)
			printf(  "\tWrote images to 'kltimg_tf*.pgm'.\n");
	}

}


