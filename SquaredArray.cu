//Input: an array of 63 numbers [0,1,2,3,…,63] 
//Output: an array of the squares of each number [0,1,4,9,…,63^2] 
/*_____________________________________________________________________________________________________*/ 

//General code: 
/*
less effecient if multiplication is expensive since it'll run 64 multiplications serially instaed of parallel
for(int i = o; i<size; i++){ 
    out[I] = in[I]*in[I]; 
} 
*/
/*_____________________________________________________________________________________________________*/ 

//CPU code: 

# include <stdio.h> 
//this is our kernel that runs on the device aka the GPU
__global__ void square(float *d_out, float *d_in){
    int idx = threadIdx.x; //each thread knows its own index
    float f = d_in[idx];
    d_out[idx] = f*f;
}
/*
notes:
    1. __global__ = declaration specifier - the way cuda knows this is a kernel
    2. each thread knows its own index withoin the block, and the index of the block it contains in all the blocks
    3. the threadIdx.x is a built-in struct for each thread. the struct contains x,y,z, where y and z are for 2D and 3D arrays of threads
    4. to launch diemntional blocks we use: 
        kernalname<<<dimentionality of blocks, diemntionality of threads per block>>>(parameters)
        the dimentionality default is 1
        kernalname<<<dim3(1,2,3), 10>>>(parameters) = 6 blocks in a box like form each containing 10 threads
        square<<<1,64>>> is equavalent to square<<<dim3(1,1,1), dim3(64,1,1)>>>
    5. kernel syntax:
        kernalname<<grid of blocks, block of threads, shared memory per block in bytes: defsult is zero>>>(params)
        example: 
        func<<<dim3(bx,by,bz), dim3(tx,ty,tz), shmem>>>(func params)
        note that the grid of blocks has max size
    6. the things threads know: (each of the following structs has x,y,z)
        threadIdx: thread index within block
        blockDim: size of a block
        blockIdx: block index withen grid
        gridDim: size of grid of blocks
    7. launching a kernel is like mapping the kernel to the elemnts where the elemnts are the threads in the blocks
        mapping is a key building block in GPU computing where we MAP(elements, function)
    */

//this is the CPU code
int main(int argc, char** argv){
    const int ARRAY_SIZE = 16;
    const int ARRAY_BYTES = ARRAY_SIZE*sizeof(float);

    //generate the input array on the host aka the CPU
    float h_in[ARRAY_SIZE];
    for(int i=0; i<ARRAY_SIZE; i++ ){
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    //declare GPU memory pointers, the pointers shuold be on the cpu but the memory allocated should be on the GPU
    float* d_in;
    float* d_out;

    //allocate GPU memory - to tell cude this is GPU memory
    cudaMalloc( (void**) &d_in, ARRAY_BYTES);
    cudaMalloc( (void**) &d_out, ARRAY_BYTES);

    //transfer the array to the GPU
    //cudeMalloc(to, from, size in bytes, direction of the transfer: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault)
    //note: cudaMemcpyDeviceToDevice moves a peice of memeory on the GPU from one place to another?
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //launch the kernel - here is where the magic happens
    //cuda launch operator is indicated using kernalname<<<number of blocks, number of threads per block>>>(parameters)
    square<<<1,ARRAY_SIZE>>>(d_out, d_in); //since ARRAY_sIZE = 64 this part launches 64 copies of square kernel on 64 threads
    /*
    notes:
        1. each nlocks is a collection of threads that exetute a kernal, the threads in each block are assigned to a single SM
        2. we can run many blocks at the same time
        3. max number of threads per block is up to 512 in older GPUs and 1024 in newer GPUs
    */
    cudaDeviceSynchronize();
    //copy back the result array to CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    //print the resulting array
    for(int i=0; i<ARRAY_SIZE; i++ ){
        printf("%f", h_out[i]);
        printf(((i%4)!=3) ? "\t" : "\n"); //to print with tabs and every 4 elemnts new line
    }

    //free the meory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0; //never forget bruv

}
