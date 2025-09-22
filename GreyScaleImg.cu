//input: a random image
//output: a grayscale of the input image

# include <stdio.h> 

/*
pixels in CUDA:
pixels in cude are represented as  a struct of uchar4:
  struct uchar4{
    unsigned char x; //red
    unsigned char y; //green
    unsigned char z; //blue
    unsigned char w; //alpha - carries the transparency information
  }
*/


/*
how to grayscale an image:
take the average of all 3 color channels multiplied by sensitivity: I = (0.299*R+0.587*G+0.114*B)/3 
*/


/*
many ways to write this in parallel, given image of size nxm:
1. we can create a grid of nm blocks where each block runs one thread that grayscales one pixel
2. we can create a grid of nm/16 where each block of 16 threads(4x4 grid) grayscales 16 pixels
3. we can create m blocks where each block of n threads grayscales one row of the image
...
*/
//this is our kernel that runs on the device aka the GPU
__global__ void rgba_to_grayscale(const uchar4* const rgbaImage, unsigned char* const greyImage, int numRows, int numCols){
//TODO
//Fill the kernel to convert from color to greyscale
//the output at each pixel should be the result of the formula above
//create a mapping from the 2D block and grid locations to an absolute 2D location in the image, then use that to calculate a 1D offset
}

void your_rgba_to_grayscale(const uchar4* const h_rgbaImage, const uchar4* const d_rgbaImage, unsigned char* const d_greyImage, size_t numRows, size_t numCols){
//you must fill in the correct sizes for the blockSize and gridSize
const dim3 blocksize(1,1,1); //TODO
const dim3 gridSize(1,1,1); //TODO
rgba_to_grayscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

cudaDeviceSynchronize();
checkCudaErrors(cudaGetLastError());

}

//input image and call kernel on each pixel
int main(int argc, char** argv){


    return 0;
}
