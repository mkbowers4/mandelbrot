# mandelbrot

example use of TBB and SSE intrinsics

usage: ./mandelbrot MODE X Y R ITER

MODE = 0 or 1. 0 will print all mandelbrot runs, 1 will only do the fastest one (sse+tbb)
    X = Xcenter
    Y = Ycenter
    R = Radius                  -- decrease to zoom
    ITER = Max# of Iterations   -- higher values increase detail, but increases runtime
    S
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
