#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

//#define __DEBUG

#define NUM_TESTS 1
#define N 4096*4096
#define THREADS_PER_BLOCK 128
#define PASS_LENGTH 5
#define ALPHABET_SIZE 81
//       ^
//       |  Make sure these match in size if changing either!
//       v
// This alphabet is based on IBM's valid password characters
const char* possibleChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~`!@#$%^&*()_-+=:;?";
const int totalBlocks = N / THREADS_PER_BLOCK;

// These never change and are pre-computed before kernel execution.
// Could have this data be #define constants, but I want this to be adaptable
// to any alphabet, password size, and thread count.
__constant__ int c_GuessesPerThread;
__constant__ long c_MaxGuesses;
__constant__ char c_Alphabet[ALPHABET_SIZE];
__constant__ char c_Password[PASS_LENGTH];

// A fast device version of string comparison. For efficiency sake, however, just
// returns 0 for non-equal, and 1 for equal. Less checks this way.
__device__ int GPU_StrCmp(const char* str1, const char* str2, unsigned size)
{
    for (unsigned i = 0; i < size; i++)
        if(str1[i] != str2[i]) return 0;

    return 1;
}

// Makes each thread compute a certain number of guesses. If any one thread's guess
// matches, throw a flag that makes all other threads stop.
__global__ void GPU_CrackKernel(int* stopFlag, long* totalGuesses)
{
#ifdef __DEBUG
    __shared__ long guessesThisBlock[THREADS_PER_BLOCK];
#endif
    // Putting data into registers since they are referred to multiple times.
    // should be faster?
    int guessesPerThread = c_GuessesPerThread;
    long maxGuesses = c_MaxGuesses;

    // Starting guess number out of maximum guess count
    long guess = (long)guessesPerThread * (threadIdx.x + (blockIdx.x * blockDim.x));

    // Same algorithm as CPU, but per-thread. Each thread will only computer a certain
    // number of guesses in parallel. 
    char guessStr[PASS_LENGTH];
    int guessesThisThread = 0;
    while(guessesThisThread < guessesPerThread && guess < maxGuesses)
    {
        long temp = guess;

        // Solving 5+ character passwords is trivial, so we
        // unroll the first 5 characters of the loop.
        guessStr[0] = c_Alphabet[temp % ALPHABET_SIZE];
        temp /= ALPHABET_SIZE;

        guessStr[1] = c_Alphabet[temp % ALPHABET_SIZE];
        temp /= ALPHABET_SIZE;

        guessStr[2] = c_Alphabet[temp % ALPHABET_SIZE];
        temp /= ALPHABET_SIZE;

        guessStr[3] = c_Alphabet[temp % ALPHABET_SIZE];
        temp /= ALPHABET_SIZE;

        guessStr[4] = c_Alphabet[temp % ALPHABET_SIZE];
        temp /= ALPHABET_SIZE;
        
        for (int i = 5; i < PASS_LENGTH; i++)
        {
            guessStr[i] = c_Alphabet[temp % ALPHABET_SIZE];
            temp /= ALPHABET_SIZE;
        }

        if (GPU_StrCmp(guessStr, c_Password, PASS_LENGTH) == 1) *stopFlag = 1;
        if (*stopFlag == 1) break;

        guess++;
        guessesThisThread++;
    }
#ifdef __DEBUG
    guessesThisBlock[threadIdx.x] = guessesThisThread;

    // Add up total number of guesses per block.
    __syncthreads();
    if (threadIdx.x == 0)
	{
        long sum = 0;
		for (int i = 0; i < THREADS_PER_BLOCK; i++)
		{
			sum += guessesThisBlock[i];
		}
		totalGuesses[blockIdx.x] = sum;
	}
#endif
}

void CreateRandomPassword(int length, char* pass)
{
    srand(time(0));
    for(int i = 0; i < length; i++)
        pass[i] = possibleChars[rand() % ALPHABET_SIZE];

    printf("\nPassword: %s\n", pass);
}

long long TimevalToMilliseconds(struct timeval tv)
{
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// Sequentially tries every possible combination of alpha-numeric characters
// and symbols until the alloted time halted or it solves the password.
// We assume password will NOT be solved. We just want to know how
// many guesses it takes the CPU in the same time the GPU took to solve it!
int CPU_CrackPassword(const char* password)
{
    struct timeval end;
    struct timeval start;
    gettimeofday(&start, NULL);

    printf("\nStarting CPU crack...\n");

    char guessStr[PASS_LENGTH];
    long totalGuesses = 1;
    for (int i = 0; i < PASS_LENGTH; i++)
    {
        totalGuesses *= ALPHABET_SIZE;
        guessStr[i] = possibleChars[0];
    }

    long guess;
    for (guess = 0; guess < totalGuesses; guess++)
    {
        // Chars go to next if the previous one flipped from the last to first
        // element. Number of guesses required to set character to next increases
        // in order of magnitude (alphabetSize)^n for each character n.
        long temp = guess;
        for (int i = 0; i < PASS_LENGTH; i++)
        {
            guessStr[i] = possibleChars[temp % ALPHABET_SIZE];
            temp /= ALPHABET_SIZE;
        }

        //printf("Guess: %ld Password: %s\n", guess, guessStr);

        if (memcmp(guessStr, password, PASS_LENGTH) == 0)
        {
            printf("Password found on CPU!\n");
            break;
        }
    }

    gettimeofday(&end, NULL);
    float milliseconds = (TimevalToMilliseconds(end) - TimevalToMilliseconds(start)) / 1000;

    printf("CPU computed %ld guesses in %.5f seconds\n", guess, milliseconds / 1000);
    return milliseconds;
}

float GPU_CrackPassword(const char* pass)
{
    printf("\nStarting GPU crack...\n");

    // Host vars.
    long *totalGuesses;
    long *maxGuesses;
    int *guessesPerThread;
    char *alphabet;
    char *password;
    
    long *d_totalGuesses;
    int *d_foundFlag;

    cudaEvent_t start, stop;    // Cuda flags for timing.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Global vars & relevant host vars.
    cudaMalloc(((void**)&d_foundFlag), sizeof(int));
    cudaMalloc(((void**)&d_totalGuesses), sizeof(long) * totalBlocks);
    
    totalGuesses = (long*)malloc(sizeof(long) * totalBlocks);
    maxGuesses = (long*)malloc(sizeof(long));
    guessesPerThread = (int*)malloc(sizeof(int));
    alphabet = (char*)malloc(sizeof(char) * ALPHABET_SIZE);
    password = (char*)malloc(sizeof(char) * PASS_LENGTH);

    *maxGuesses = (long)pow(ALPHABET_SIZE, PASS_LENGTH);
    *guessesPerThread = (int)ceil((double)*maxGuesses / (N));

    for(int i = 0; i < ALPHABET_SIZE; i++)
        alphabet[i] = possibleChars[i];
    for(int i = 0; i < PASS_LENGTH; i++)
        password[i] = pass[i];

    cudaMemcpyToSymbol(c_MaxGuesses, maxGuesses, sizeof(long));
    cudaMemcpyToSymbol(c_GuessesPerThread, guessesPerThread, sizeof(int));
    cudaMemcpyToSymbol(c_Alphabet, alphabet, sizeof(char) * ALPHABET_SIZE);
    cudaMemcpyToSymbol(c_Password, password, sizeof(char) * PASS_LENGTH);

    printf("Maximum guesses to compute: %ld\nGuesses per GPU thread: %d\n", *maxGuesses, *guessesPerThread);

    // Call kernel + time with events.
    cudaEventRecord(start);

    GPU_CrackKernel<<< totalBlocks, THREADS_PER_BLOCK >>>(d_foundFlag, d_totalGuesses);

    cudaEventRecord(stop);

    // Get total guesses, calculate time it took to solve.
    cudaMemcpy(totalGuesses, d_totalGuesses, sizeof(long) * totalBlocks, cudaMemcpyDeviceToHost);
    long sum = 0;
    for (int i = 0; i < totalBlocks; i++)
        sum += totalGuesses[i];

    if (sum > 0)
        printf("Total Guesses made: %ld\n", sum);
    printf("\n");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU found password in: %.5f sec\n", (milliseconds / 1000));

    free(totalGuesses);
    free(maxGuesses);
    free(guessesPerThread);
    free(alphabet);
    free(password);

    cudaFree(d_totalGuesses);
    cudaFree(d_foundFlag);

    return milliseconds;
}

int main(int argc, char *argv[])
{
    // Check for running CPU flag.
    int runCPU = 0;
    if (argc == 2)
        if (!strcmp(argv[1], "-cpu")) runCPU = 1;

#ifdef __DEBUG
    FILE *fp;
    fp = fopen("Results.txt", "w+");
#endif
    char* password = (char*)malloc(sizeof(char) * PASS_LENGTH);

    // Call CPU/GPU functions specified number of times with random password.
    for (int i = 0; i < NUM_TESTS; i++)
    {
        CreateRandomPassword(PASS_LENGTH, password);

        float gpuBenchmark = GPU_CrackPassword(password);
        if (runCPU == 1)
            float cpuBenchmark = CPU_CrackPassword(password);
        
        printf("\n******\n");

#ifdef __DEBUG
        fprintf(fp, "%.5f\n", gpuBenchmark / 1000);
#endif
    }

    free(password);
    return -1;
}