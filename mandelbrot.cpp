/*
 * Fall 2021
 * Michael Bowers
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "tbb/tbb.h"
#define MandelbrotSIZE (1024)  //length&width of mandelbrot. Keep this a multiple of 4

#ifdef __SSE2__
  #include <emmintrin.h>
#else
  #warning No SSE support! Will not compile. // will yell if missing -msse2 in makefile
#endif

/* usage: ./mandelbrot MODE X Y R ITER
    
    MODE = 0 or 1. 0 will print all mandelbrot runs, 1 will only do the fastest one (sse+tbb)
    X = Xcenter  
    Y = Ycenter  
    R = Radius                  -- decrease to zoom 
    ITER = Max# of Iterations   -- higher values increase detail, but increases runtime
    
    example coordinates:
        ./mandelbrot 0 0 0 2 256                                      -> standard "home screen" view
        ./mandelbrot 0 -0.7568097768953 -0.066879524349399 0.001 2048 -> swirls 
        ./mandelbrot 0 0.380 0.31 0.005 512                           -> a large swirl somewhere above the main mandelbrot 
        ./mandelbrot 0 0.318 0.5 0.005 4096                           -> "fingers"
        ./mandelbrot 0 -1.94153391 0.000129404 1 1024                 -> adjust the radius down to find a baby mandelbrot
        ./mandelbrot 0 0.30018 0.4618 0.0001, 2048                    -> Hidden Mandelbrot
        ./mandelbrot 0 -0.21757193676756 1.114419679 0.01 256         -> lightning. Zoom in for a baby brot
        ./mandelbrot 0 -0.75 0.25 0.06 256                            -> "seahorses"
        ./mandelbrot 0 -0.811531 0.201429 0.001 256                   -> "octopus"
        ./mandelbrot 0 0.2969 0.020 0.015 1024                        -> "elephants"
        ./mandelbrot 0 -1.25066 0.02012 0.0001 1024                   -> eye spirals
        ./mandelbrot 0 -0.235125 0.827215 0.0001 1024                 -> many branches
        ./mandelbrot 0 0.254987 -0.00056797 0.0001 2048               -> two opposing spirals
        ./mandelbrot 0 -0.745428 0.113009 0.0001 1024                 -> circular viral-looking structures
        ./mandelbrot 0 -1.2535 -0.0467 0.001 1024                     -> a region my Grandfather found 30 yrs ago with his mandelbrot explorer
                                                                                                                        (his original slide in ppt)
         
    Notes: 
        The Mandelbrot set is the set of complex numbers "c" for which the equation F(z) = z^2 + c iterated from n = 0 does not diverge to infinity.
        
        Navigating the Mandelbrot is a bit tricky, because the interesting regions reside very tightly around the Mandelbrot boundaries, 
        and as we zoom in further we have to be more careful when adjusting Xcenter and Ycenter (need to change x&y more finely).
                
        There is a good open-source program called Fraqtive (https://fraqtive.mimec.org/) that is very easily navigable with the mouse.
        It is much easier to find coordinates, which can then be plugged into this Mandelbrot generator. 
        The coordinate system in this generator happen to be the same as the one in Fraqtive, where increasing y moves the view down, 
        and increasing x moves to the right.
        
        If iteration values of 256, 512, 1024, 2048, etc. are used, the main mandelbrot will be black for any color palette
        if using values such as 255, 511, 1023, etc., the main mandelbrot will be white (seems to only work when using a grayscale color scheme)
        This way, it's much easier to visualize the boundary of the set.
        
        I achieved some very interesting effects by changing the threshold checking in the main mandelbrot loop
        Instead of the original (xsqr+ysqr) < 4, changing the left to any one of these will produce some weird/cool radiating patterns:
                (zx*zy), (xsqr+xsqr), (ysqr+ysqr), (xsqr*ysqr), (zx*ysqr), (zy*xsqr), (zx*xsqr), (zy*ysqr), (zx+zy), (zx*(zy/(k+1))), (xsqr*(zy/(k+1))),
                or perhaps many other variations of such.
        Somehow, the boundaries of the mandelbrot remain the same, but how the iteration values radiate away from the mandelbrot
        is wildly different, and makes for some cool variations of the original mandelbrot!
        
        I found that the most effective way to verify the integrity of the SSE calculations is to compare checksums of a mandelbrot generated 
        via SSE to a mandelbrot from a non-SSE run. I did this with the md5sum tool, and by having all my brots follow the same exact 
        color scheme (I did a grayscale for visual simplicity). For regular grayscale mandelbrots (where (xsqr+ysqr) < 4), checksums showed identical results.
        However, if I follow the same procedure for the (zx*zy) < 4 variant and compare SSE vs non-SSE, the results are 
        slightly different both visually and via checksum. TODO: Figure out why this is the case. 
*/

typedef struct {
   uint32_t *pixarr_s;  //sequential data
   uint32_t *pixarr_p;  //parallel data
   uint32_t *pixarr_v;  //vectored data
   uint32_t *pixarr_vp; //parallel vectored data
   int width;
   int height;
} mand_args;

//prototypes
__m128i SSE_GetIterations(__m128 x, __m128 y, uint32_t i);
uint32_t GetIterations(int i, int j, float *y, float *x, uint32_t max);
void ParallelBrot(float *x, float *y, mand_args MB, int maxiterations);
void SequentialBrot(float *x, float *y, mand_args MB, int maxiterations);
void SSEBrot(float *x, float *y, mand_args MB, int maxiterations);
void TBBSSEBrot(float *x, float *y, mand_args MB, int maxiterations, int j);
__m128i SSE_GetIterations(__m128 x, __m128 y, uint32_t i);



int main (int argc, char *argv[]) {
    int mode = 0;
    int i;
    float *xarr, *yarr;
    mand_args MB;   //struct with all the mandelbrot pixel array results
   
    struct timeval start, end, start_p, end_p, start_v, end_v, start_vp, end_vp;
    long t_us, t_us_p, t_us_v, t_us_vp;
    
        //these dimensions should be divisible by 4 so we don't run into 
        //issues with our vector-forming, which takes in 4 points at a time
    MB.width = MandelbrotSIZE;   
    MB.height = MandelbrotSIZE;  
    
    int SIZE = MB.width*MB.height;
    xarr = (float*)calloc(MB.width, sizeof(float));
    yarr = (float*)calloc(MB.height, sizeof(float));
    MB.pixarr_s  = (uint32_t*)malloc(SIZE*sizeof(uint32_t));
    MB.pixarr_p  = (uint32_t*)malloc(SIZE*sizeof(uint32_t));
    MB.pixarr_v  = (uint32_t*)malloc(SIZE*sizeof(uint32_t));
    MB.pixarr_vp = (uint32_t*)malloc(SIZE*sizeof(uint32_t));
    
    if (argc != 6)
    {
        printf("Follow Usage: ./mandelbrot MODE X  Y  R  ITER\n");
        printf("Try this out! ./mandelbrot MODE 0  0  2  256\n");
	 printf("MODE = 0 for all brots. MODE = 1 for fastest brot\n");
        exit(-1);
    }
    
    char *argptr;
    mode = atoi(argv[1]);
    const float ycenter = strtof(argv[3], &argptr);
    const float radius = strtof(argv[4], &argptr);
    const float xcenter = strtof(argv[2], &argptr);
    const int maxiterations = atoi(argv[5]);
    const float xscale = radius/(MB.width / 2);
    const float yscale = radius/(MB.height / 2);
    
    if (mode != 0 && mode != 1)
    {
      printf("ERROR: incorrect mode\n");
      printf("MODE = 0 for all brots. MODE = 1 for fastest brot (SSE+tbb)\n");
      exit(-1);
    }
    
    //creating x and y arrays. Each mandelbrot run will use the same x and y data
    //the x and y point data get scaled around the specified center coordinates
    for (i=0; i<MB.height; i++)
        { yarr[i] = ycenter - radius + (i*yscale); }
    for (i=0; i<MB.width; i++) 
        { xarr[i] = xcenter - radius + (i*xscale); }
    
    
    if (mode == 0)
    {
        //////////////////////////////
        //////  SEQUENTIAL RUN  //////
        //////////////////////////////
        gettimeofday(&start, NULL);
        SequentialBrot(xarr, yarr, MB, maxiterations);
        gettimeofday(&end, NULL);
        
        //////////////////////////////
        //////   PARALLEL RUN   //////
        //////////////////////////////
        gettimeofday(&start_p, NULL);
        ParallelBrot(xarr, yarr, MB, maxiterations);
        gettimeofday(&end_p, NULL);
        
        //////////////////////////////
        //////  SIMD (SSE) RUN  //////
        //////////////////////////////
        gettimeofday(&start_v, NULL);
        SSEBrot(xarr, yarr, MB, maxiterations);
        gettimeofday(&end_v, NULL);
    }
    
    /////////////////////////////
    //////   SSE+TBB RUN   //////
    /////////////////////////////
    gettimeofday(&start_vp, NULL);
    tbb::parallel_for (int(0), int(MB.height), [&] (int j) {
      TBBSSEBrot(xarr, yarr, MB, maxiterations, j);
    });
    gettimeofday(&end_vp, NULL);
    

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //  With coloring, every iteration # gets colored differently. We have a multiplier, we can  //
    //  mask some bits, and we can shift the bits to change to change colors. It's not a very    //
    //  precise science, but we can surely bias the colors towards a particular palette          //
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    if (mode == 0) // if 0 is selected we run through the first 3. If one, we skip these and just do the fastest brot.
    {
        ///////////////////////////////////////
        // generating sequential brot colors //
        ///////////////////////////////////////
        
        
        //change to grayscale changing all masks to 0xff, all right shifts to 0. cfactor = 1 for very plain mandelbrot
        
        FILE *fp_s = fopen("sbrot.ppm", "wb"); /* b - binary mode */
        (void) fprintf(fp_s, "P6 %d %d 255\n", MB.width, MB.height); //file stream, P6 for binary, both dimensions of file, 255 is max value
        uint8_t cfactor_s = 100; // This is a multiplier to make pretty colors
        uint8_t color_s[3];
        for(i=0; i < SIZE; i++){
            color_s[2] = ((MB.pixarr_s[i] & 0xff) >> 0)*cfactor_s; //b   
            color_s[1] = ((MB.pixarr_s[i] & 0xff) >> 0)*cfactor_s; //g
            color_s[0] = ((MB.pixarr_s[i] & 0xff) >> 0)*cfactor_s; //r
            (void) fwrite(color_s, 1, 3, fp_s);
                    //fwrite(ptr, size, count, stream)
        }
        (void) fclose(fp_s);
        
        /////////////////////////////////////
        // generating parallel brot colors //
        /////////////////////////////////////
        FILE *fp_p = fopen("pbrot.ppm", "wb"); /* b - binary mode */
        (void) fprintf(fp_p, "P6 %d %d 255\n", MB.width, MB.height);
        uint8_t cfactor_p = 100; // This is a multiplier to make pretty colors
        uint8_t color_p[3];
        for(i=0; i < SIZE; i++){
            color_p[2] = ((MB.pixarr_p[i] & 0xff) >> 0)*cfactor_p; //b   
            color_p[1] = ((MB.pixarr_p[i] & 0xff) >> 0)*cfactor_p; //g
            color_p[0] = ((MB.pixarr_p[i] & 0xff) >> 0)*cfactor_p; //r
            (void) fwrite(color_p, 1, 3, fp_p);
                    //fwrite(ptr, size, count, stream)
        }
        (void) fclose(fp_p);
        
        ////////////////////////////////
        // generating sse brot colors //
        ////////////////////////////////
        FILE *fp_v = fopen("vbrot.ppm", "wb"); /* b - binary mode */
        (void) fprintf(fp_v, "P6 %d %d 255\n", MB.width, MB.height);
        uint8_t cfactor_v = 100; // This is a multiplier to make pretty colors
        uint8_t color_v[3];
        for(i=0; i < SIZE; i++){
            color_v[2] =  ((MB.pixarr_v[i] & 0xff) >> 0)*cfactor_v; //b  
            color_v[1] =  ((MB.pixarr_v[i] & 0xff) >> 0)*cfactor_v; //g
            color_v[0] =  ((MB.pixarr_v[i] & 0xff) >> 0)*cfactor_v; //r
            (void) fwrite(color_v, 1, 3, fp_v);
        }
        (void) fclose(fp_v);
    }
    
    ////////////////////////////////////////////////////////
    // generating parallel vectored (tbb+see) brot colors //
    ////////////////////////////////////////////////////////
    FILE *fp_vp = fopen("vpbrot.ppm", "wb"); // b - binary mode 
    (void) fprintf(fp_vp, "P6 %d %d 255\n", MB.width, MB.height);
    uint8_t cfactor_vp = 100; // This is a multiplier to make pretty colors
    uint8_t color_vp[3];
    for(i=0; i < SIZE; i++){
        color_vp[2] = ((MB.pixarr_vp[i] & 0xff) >> 0)*cfactor_vp; //b  
        color_vp[1] = ((MB.pixarr_vp[i] & 0xff) >> 0)*cfactor_vp; //g       
        color_vp[0] = ((MB.pixarr_vp[i] & 0xff) >> 0)*cfactor_vp; //r
        (void) fwrite(color_vp, 1, 3, fp_vp);
    }
    (void) fclose(fp_vp); 
    
    if (mode == 0) //if '0' is selected we run through the first 3 and then move onto the last one. If '1' we skip these and just do the fastest brot.
    {
        //PRINTING TIME MEASUREMENTS (SEQUENTIAL)
        printf ("Sequential Run:\n");
        printf ("	start: %ld us\n", start.tv_usec); // start.tv_sec
        printf ("	end: %ld us\n", end.tv_usec);    // end.tv_sec; 
        t_us = (end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec; // for ms: define t_ms as double and divide by 1000.0
        // gettimeofday: returns current time. So, when the secs increment, the us resets to 0.
        printf ("	Elapsed time: %ld us\n", t_us); 
        
        //PRINTING TIME MEASUREMENTS (TBB)
        printf ("Parallel(tbb) Run:\n");
        printf ("	start: %ld us\n", start_p.tv_usec); // start.tv_sec
        printf ("	end: %ld us\n", end_p.tv_usec);    // end.tv_sec; 
        t_us_p = (end_p.tv_sec - start_p.tv_sec)*1000000 + end_p.tv_usec - start_p.tv_usec; // for ms: define t_ms as double and divide by 1000.0
        // gettimeofday: returns current time. So, when the secs increment, the us resets to 0.
        printf ("	Elapsed time: %ld us\n", t_us_p);
        
        //PRINTING TIME MEASUREMENTS (SSE)
        printf ("Vectored(sse) Run:\n");
        printf ("	start: %ld us\n", start_v.tv_usec); // start.tv_sec
        printf ("	end: %ld us\n", end_v.tv_usec);    // end.tv_sec; 
        t_us_v = (end_v.tv_sec - start_v.tv_sec)*1000000 + end_v.tv_usec - start_v.tv_usec; // for ms: define t_ms as double and divide by 1000.0
        // gettimeofday: returns current time. So, when the secs increment, the us resets to 0.
        printf ("	Elapsed time: %ld us\n", t_us_v);
    }
    
    //PRINTING TIME MEASUREMENTS (Paralleled Vectored (SSE & tbb)
    printf ("TBB & SSE Run:\n");
    printf ("	start: %ld us\n", start_vp.tv_usec); // start.tv_sec
    printf ("	end: %ld us\n", end_vp.tv_usec);    // end.tv_sec; 
    t_us_vp = (end_vp.tv_sec - start_vp.tv_sec)*1000000 + end_vp.tv_usec - start_vp.tv_usec; // for ms: define t_ms as double and divide by 1000.0
     // gettimeofday: returns current time. So, when the secs increment, the us resets to 0.
    printf ("	Elapsed time: %ld us\n", t_us_vp);
    
    free(xarr); free(yarr); free(MB.pixarr_s); free(MB.pixarr_p); 
    free(MB.pixarr_v); free(MB.pixarr_vp);
}

    


uint32_t GetIterations(int i, int j, float *y, float *x, uint32_t max) {   //getting iterations for non-sse versions
    uint32_t k=0;                //k is iteration number
    float zx=0.000, zy=0.000, xsqr=0.000, ysqr=0.000;
    float cx = x[i], cy = y[j];  //grab initial values

    //main mandelbrot loop.
        //there may be a way to write this so the compiler pipelines the instructions better.
    while (((xsqr+ysqr)< 4) && (k < max))  //change (xsqr+ysqr) to some of the variations in the top comments instead for some interesting effects!
    {
        zy = 2*zx*zy + cy;
        zx = xsqr - ysqr + cx;
        xsqr = zx * zx;
        ysqr = zy * zy;
        k++;
    }
    return k; //iterations
}
void ParallelBrot(float *x, float *y, mand_args MB, int maxiterations) {
    int width, height;
    uint32_t max = (uint32_t)maxiterations;
    uint32_t *pix;
    width = MB.width; height = MB.height; pix = MB.pixarr_p;
    tbb::parallel_for (int(0), int(height), [&] (int j)
    {
        tbb::parallel_for (int(0), int(width), [&] (int i){
            pix[i+(height*j)] = GetIterations(i, j, y, x, max);
        });
    });
}

void SequentialBrot(float *x, float *y, mand_args MB, int maxiterations) {
    int i, j, width, height;
    uint32_t max = (uint32_t)maxiterations;
    uint32_t *pix;
    width = MB.width; height = MB.height; pix = MB.pixarr_s;
    for (j=0; j<height; j++){
        for (i=0; i<width; i++)
        {
            pix[i+(height*j)] = GetIterations(i, j, y, x, max);
        }
    }
}

void SSEBrot(float *x, float *y, mand_args MB, int maxiterations) {
    int i, width, height;
    uint32_t max = (uint32_t)maxiterations; uint32_t *pix;
    width = MB.width; height = MB.height; pix = MB.pixarr_v;

    for (int j=0; j<height; j++)
    {
      __m128i ans = _mm_set_epi32(0,0,0,0);     //initialize answer placeholder
      __m128 sse_y = _mm_set1_ps(*(y+j));       //put in y value of current row
        for (i=0; i<width; i+=4)
        {
            __m128 sse_x = _mm_set_ps(*(x+i),*(x+i+1),*(x+i+2),*(x+i+3));  //put in the next 4 x values. Basically sliding across the row
            ans = SSE_GetIterations(sse_x, sse_y, max);                    //ans vector contains iteration values

            pix[i+(height*j)]   = _mm_extract_epi16(ans,6);    //_mm_extract_epi16(ans, n) where n ranges from 0 to 7
            pix[i+1+(height*j)] = _mm_extract_epi16(ans,4);    // n is the index for the 16-bit int we extract from the 128-bit result
            pix[i+2+(height*j)] = _mm_extract_epi16(ans,2);    // need to make sure the order/orientation matches the order in which we loaded sse_x
            pix[i+3+(height*j)] = _mm_extract_epi16(ans,0);
        }
    }
}

void TBBSSEBrot(float *x, float *y, mand_args MB, int maxiterations, int j) {
    int i, width, height;
    uint32_t max = (uint32_t)maxiterations;
    uint32_t *pix;
    width = MB.width; height = MB.height; pix = MB.pixarr_vp;
    __m128i ans = _mm_set_epi32(0,0,0,0);    //initialize answer placeholder
    __m128 sse_y = _mm_set1_ps(*(y+j));    //put in y value of current row
    for (i=0; i<width; i+=4)
    {
        __m128 sse_x = _mm_set_ps(*(x+i),*(x+i+1),*(x+i+2),*(x+i+3)); //put in the next 4 x values. Basically sliding across the row
        ans = SSE_GetIterations(sse_x, sse_y, max);                   //ans vector contains iteration values

        pix[i+(height*j)]   = _mm_extract_epi16(ans,6);     //_mm_extract_epi16(ans, n) where n ranges from 0 to 7
        pix[i+1+(height*j)] = _mm_extract_epi16(ans,4);     // n is the index for the 16-bit int we extract from the 128-bit result
        pix[i+2+(height*j)] = _mm_extract_epi16(ans,2);     // need to make sure the order/orientation matches the order in which we loaded sse_x
        pix[i+3+(height*j)] = _mm_extract_epi16(ans,0);
    }
}



__m128i SSE_GetIterations(__m128 x, __m128 y, uint32_t i) {   //getting iterations for sse versions
    __m128 xi = x;
    __m128 yi = y;
    __m128i iterations = _mm_set_epi32(1,1,1,1);      //start initial count at 1, because the sequential always goes thru 1 iteration
    __m128 threshold = _mm_set1_ps(4);                //vector of our threshold 4.0
    __m128i ones = _mm_set_epi32(1,1,1,1);            // vector of ones
    uint32_t j=1;

    while(j < i)
    {
        // __m128  is a vector of 4x32-bit floats
        // __m128i is a vector of 4x32-bit ints

        __m128 xsqr = _mm_mul_ps(x, x);   // xsqr = x*x
        __m128 ysqr = _mm_mul_ps(y, y);   // ysqr = y*y
        __m128 xy   = _mm_mul_ps(x, y);   // xy = x*y
        __m128 xsqr_plus_ysqr = _mm_add_ps(xsqr, ysqr);     // xsqr_plus_ysqr = x*x + y*y

                        //typecasting here seems to work. There are also SSE instructions that allow conversions between m128, m128d and m128i
        __m128i check    = (__m128i)_mm_cmplt_ps(xsqr_plus_ysqr, threshold);     // 0 if done, 0xFFFFFFFF if not done
                                                                             // analogous to xsqr + ysqr > 4

        __m128i add_iter = _mm_and_si128(check, ones);                           // 0 if done, 1 if not done (converting 0xFFFFFFFF to 0x00000001)

        //mm_movemask_epi8 will create mask from the most significant bit of each 8-bit element of the input vector
        //and store result in an int (16-bit)

        if (_mm_movemask_epi8(check) == 0x0000)     // here we wait until ALL points are done.
        {                                           // "check" should be all zeros if we're done (MSBs of each 8 bit slice should be zero)
            return iterations;
        }
        else                                        // not all done
        {
            iterations = _mm_add_epi32(iterations, add_iter);     //increment points that need to be incremented
            x = _mm_add_ps(_mm_sub_ps(xsqr,ysqr), xi);            //x = x^2 - y^2 + xi
            y = _mm_add_ps(_mm_add_ps(xy,xy), yi);                //y = 2xy + yi
            j++;
        }
    }
    return iterations;
}

