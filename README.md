# mandelbrot

example use of TBB and SSE intrinsics

usage: ./mandelbrot MODE X Y R ITER

MODE = 0 or 1. 0 for all runs, 1 for fastest
X = Xcenter
Y = Ycenter
R = Radius                  
ITER = Max# of Iterations   
    
example coordinates:
        ./mandelbrot 0 0 0 2 256                                    
        ./mandelbrot 0 -0.7568097 -0.0668795 0.001 2048 
        ./mandelbrot 0 0.380 0.31 0.005 512                          
        ./mandelbrot 0 0.318 0.5 0.005 4096         
        ./mandelbrot 0 -1.94153391 0.000129404 1 1024        
        ./mandelbrot 0 0.30018 0.4618 0.0001, 2048     
        ./mandelbrot 0 -0.2175719 1.1144196 0.01 256    
        ./mandelbrot 0 -0.75 0.25 0.06 256                
        ./mandelbrot 0 -0.811531 0.201429 0.001 256     
        ./mandelbrot 0 0.2969 0.020 0.015 1024       
        ./mandelbrot 0 -1.25066 0.02012 0.0001 1024      
        ./mandelbrot 0 -0.235125 0.827215 0.0001 1024     
        ./mandelbrot 0 0.254987 -0.00056797 0.0001 2048           
        ./mandelbrot 0 -0.745428 0.113009 0.0001 1024     
        ./mandelbrot 0 -1.2535 -0.0467 0.001 1024        
