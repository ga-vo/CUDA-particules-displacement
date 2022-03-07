# CUDA-particules-displacement

## Compile
Use  `nvcc -arch=<GPU_ARCHITECTURE_CODE> -ccbin g++ -Xcompiler="-lboost_iostreams" -o fluido fluido.cu`

You can check the *GPU_ARCHITECTURE_CODE* corresponding to your GPU architecture <a href="https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/" target="_blank">here</a>

Must have *libgnuplot-iostream-dev* and *CUDA* (obviously xd) installed.
<br>
<br>
## RUN 
Use `./fluido` with default parameters

and `./fluido -np <value:int> -dt <value:double> -tmax <value:int>` using custom parameters
<br>
<br>
##

### Using GNUPlot c++ Library 
  * https://github.com/dstahlke/gnuplot-iostream
  * http://stahlke.org/dan/gnuplot-iostream/
