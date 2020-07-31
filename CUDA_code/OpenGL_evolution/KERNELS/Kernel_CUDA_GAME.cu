typedef unsigned char ubyte;


__global__ void simpleLifeKernel(const ubyte* lifeData, uint worldWidth, uint worldHeight, ubyte* resultLifeData){

	uint worldSize = worldWidth * worldHeight;

	for(unit cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; cellId < worldSize; cellId += blockDim.x * gridDim.x){

		uint x = cellId % worldSize;
		uint yAbs = cellId  - x;
		uint xLeft = (x + worldWidth - 1) % worldWidth;
		uint xRight = (x + 1) % worldWidth;
		uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
    	uint yAbsDown = (yAbs + worldWidth) % worldSize;

    	uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
      				+ lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
      				+ lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

      	resultLifeData[x + yAbs] = aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;

	}
}


__global__ void bitLifeKernelNoLookup(const ubyte* lifeData, uint worldDataWidth, uint worldHeight, uint bytesPerThread, ubyte* resultLifeData) {
 
  uint worldSize = (worldDataWidth * worldHeight);
 
  	for (uint cellId = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * bytesPerThread; cellId < worldSize; cellId += blockDim.x * gridDim.x * bytesPerThread) {
 
    	uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
    	uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
    	uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
    	uint yAbsDown = (yAbs + worldDataWidth) % worldSize;
 
    	// Initialize data with previous byte and current byte.
    	uint data0 = (uint)lifeData[x + yAbsUp] << 16;
    	uint data1 = (uint)lifeData[x + yAbs] << 16;
    	uint data2 = (uint)lifeData[x + yAbsDown] << 16;
 
    	x = (x + 1) % worldDataWidth;
    	data0 |= (uint)lifeData[x + yAbsUp] << 8;
    	data1 |= (uint)lifeData[x + yAbs] << 8;
    	data2 |= (uint)lifeData[x + yAbsDown] << 8;
 
    	for (uint i = 0; i < bytesPerThread; ++i) {
      		uint oldX = x;  // old x is referring to current center cell
      		x = (x + 1) % worldDataWidth;
      		data0 |= (uint)lifeData[x + yAbsUp];
      		data1 |= (uint)lifeData[x + yAbs];
      		data2 |= (uint)lifeData[x + yAbsDown];
 
      		uint result = 0;
      		for (uint j = 0; j < 8; ++j) {
        		uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
        		aliveCells >>= 14;
        		aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u) + ((data2 >> 15) & 0x1u);
 				result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);
 
        		data0 <<= 1;
        		data1 <<= 1;
        		data2 <<= 1;
      		}
 
      		resultLifeData[oldX + yAbs] = result;
    	}
  	}
}


__global__ void bitLifeCountingBigChunks(const uint* lifeData, uint worldDataWidth, uint worldHeight, uint chunksPerThread, uint* resultLifeData) {
 
  	uint worldSize = (worldDataWidth * worldHeight);
 
  	for (uint cellId = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * chunksPerThread; cellId < worldSize; cellId += blockDim.x * gridDim.x * chunksPerThread) {
 
    	uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
    	uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
    	uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
    	uint yAbsDown = (yAbs + worldDataWidth) % worldSize;
 
    	uint currData0 = swapEndianessUint32(lifeData[x + yAbsUp]);
    	uint currData1 = swapEndianessUint32(lifeData[x + yAbs]);
    	uint currData2 = swapEndianessUint32(lifeData[x + yAbsDown]);
 
    	x = (x + 1) % worldDataWidth;
    	uint nextData0 = swapEndianessUint32(lifeData[x + yAbsUp]);
    	uint nextData1 = swapEndianessUint32(lifeData[x + yAbs]);
    	uint nextData2 = swapEndianessUint32(lifeData[x + yAbsDown]);
 
    	for (uint i = 0; i < chunksPerThread; ++i) {
      		// Evaluate front overlapping cell.
      		uint aliveCells = (currData0 & 0x1u) + (currData1 & 0x1u) + (currData2 & 0x1u)
        			+ (nextData0 >> 31) + (nextData2 >> 31)  // Do not count middle cell.
        			+ ((nextData0 >> 30) & 0x1u) + ((nextData1 >> 15) & 0x1u)
        			+ ((nextData2 >> 16) & 0x1u);
 
      		// 31-st bit.
      		uint result = (aliveCells == 3 || (aliveCells == 2 && (nextData1 >> 31)))? (1u << 31) : 0u;
 
      		uint oldX = x;  // Old x is referring to current center cell.
      		x = (x + 1) % worldDataWidth;
      		currData0 = nextData0;
      		currData1 = nextData1;
      		currData2 = nextData2;
 
      		nextData0 = swapEndianessUint32(lifeData[x + yAbsUp]);
      		nextData1 = swapEndianessUint32(lifeData[x + yAbs]);
      		nextData2 = swapEndianessUint32(lifeData[x + yAbsDown]);
 
      		// Evaluate back overlapping cell.
      		aliveCells = ((currData0 >> 1) & 0x1u) + ((currData1 >> 1) & 0x1u)
        		+ ((currData2 >> 1) & 0x1u)
        		+ (currData0 & 0x1u) + (currData2 & 0x1u)  // Do not count middle cell.
        		+ (nextData0 >> 31) + (nextData1 >> 31) + (nextData2 >> 31);
 
      		// 0-th bit.
      		result |= (aliveCells == 3 || (aliveCells == 2 && (currData1 & 0x1u))) ? 1u : 0u;
 
      		// The middle cells with no overlap.
      		for (uint j = 0; j < 30; ++j) {
        	uint shiftedData = currData0 >> j;
        	uint aliveCells = (shiftedData & 0x1u) + ((shiftedData >> 1) & 0x1u) + ((shiftedData >> 2) & 0x1u);
 
        	shiftedData = currData2 >> j;
        	aliveCells += (shiftedData & 0x1u) + ((shiftedData >> 1) & 0x1u) + ((shiftedData >> 2) & 0x1u);
 
        	shiftedData = currData1 >> j;
        	// Do not count middle cell.
        	aliveCells += (shiftedData & 0x1u) + ((shiftedData >> 2) & 0x1u);
 
        	result |= (aliveCells == 3 || (aliveCells == 2 && (shiftedData & 0x2))? (2u << j) : 0u);
      	}
 
      	resultLifeData[oldX + yAbs] = swapEndianessUint32(result);
    }
  }
}



__global__ void precompute6x3EvaluationTableKernel(ubyte* resultEvalTableData) {
  	
  	uint tableIndex = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
 
  	ubyte resultState = 0;
  	// For each cell.
  	for (uint dx = 0; dx < 4; ++dx) {
    	// Count alive neighbors.
    	uint aliveCount = 0;
    	for (uint x = 0; x < 3; ++x) {
      		for (uint y = 0; y < 3; ++y) {
        		aliveCount += getCellState(x + dx, y, tableIndex);
      		}
    	}
 
    	uint centerState = getCellState(1 + dx, 1, tableIndex);
    	aliveCount -= centerState;  // Do not count center cell in the sum.
 
    	if (aliveCount == 3 || (aliveCount == 2 && centerState == 1)) {
     		resultState |= 1 << (3 - dx);
    	}
  	}
 
 	resultEvalTableData[tableIndex] = resultState;
}


inline __device__ uint getCellState(uint x, uint y, uint key) {
  uint index = y * 6 + x;
  return (key >> ((3 * 6 - 1) - index)) & 0x1;
}


__global__ void bitLifeKernel(const ubyte* lifeData, uint worldDataWidth,
    uint worldHeight, uint bytesPerThread, const ubyte* evalTableData,
    ubyte* resultLifeData) {
  uint worldSize = (worldDataWidth * worldHeight);
 
  for (uint cellId = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * bytesPerThread;
      cellId < worldSize;
      cellId += blockDim.x * gridDim.x * bytesPerThread) {
 
    uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // start at block x - 1
    uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
    uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
    uint yAbsDown = (yAbs + worldDataWidth) % worldSize;
 
    // Initialize data with previous byte and current byte.
    uint data0 = (uint)lifeData[x + yAbsUp] << 8;
    uint data1 = (uint)lifeData[x + yAbs] << 8;
    uint data2 = (uint)lifeData[x + yAbsDown] << 8;
 
    x = (x + 1) % worldDataWidth;
    data0 |= (uint)lifeData[x + yAbsUp];
    data1 |= (uint)lifeData[x + yAbs];
    data2 |= (uint)lifeData[x + yAbsDown];
 
    for (uint i = 0; i < bytesPerThread; ++i) {
      uint oldX = x;  // Old x is referring to current center cell.
      x = (x + 1) % worldDataWidth;
      data0 = (data0 << 8) | (uint)lifeData[x + yAbsUp];
      data1 = (data1 << 8) | (uint)lifeData[x + yAbs];
      data2 = (data2 << 8) | (uint)lifeData[x + yAbsDown];
 
      uint lifeStateHi = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
      uint lifeStateLo = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);
 
      resultLifeData[oldX + yAbs] = (evalTableData[lifeStateHi] << 4) | evalTableData[lifeStateLo];
    }
  }
}



bool runBitLifeKernel(ubyte*& d_encodedLifeData, ubyte*& d_encodedlifeDataBuffer,
    const ubyte* d_lookupTable, size_t worldWidth, size_t worldHeight,
    size_t iterationsCount, ushort threadsCount, uint bytesPerThread,
    bool useBigChunks) {
 
  // World has to fit into 8 bits of every byte exactly.
  if (worldWidth % 8 != 0) {
    return false;
  }
 
  size_t worldEncDataWidth = worldWidth / 8;
  if (d_lookupTable == nullptr && useBigChunks) {
    size_t factor = sizeof(uint) / sizeof(ubyte);
    if (factor != 4) {
      return false;
    }
 
    if (worldEncDataWidth % factor != 0) {
      return false;
    }
    worldEncDataWidth /= factor;
  }
 
  if (worldEncDataWidth % bytesPerThread != 0) {
    return false;
  }
 
  size_t encWorldSize = worldEncDataWidth * worldHeight;
 
  if ((encWorldSize / bytesPerThread) % threadsCount != 0) {
    return false;
  }
 
  size_t reqBlocksCount = (encWorldSize / bytesPerThread) / threadsCount;
  ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);
 
  if (d_lookupTable == nullptr) {
    if (useBigChunks) {
      uint*& data = (uint*&)d_encodedLifeData;
      uint*& result = (uint*&)d_encodedlifeDataBuffer;
 
      for (size_t i = 0; i < iterationsCount; ++i) {
        bitLifeKernelCountingBigChunks<<<blocksCount, threadsCount>>>(data,
          uint(worldEncDataWidth), uint(worldHeight), bytesPerThread, result);
        std::swap(data, result);
      }
    }
    else {
      for (size_t i = 0; i < iterationsCount; ++i) {
        bitLifeKernelCounting<<<blocksCount, threadsCount>>>(d_encodedLifeData,
          uint(worldEncDataWidth), uint(worldHeight), bytesPerThread,
          d_encodedlifeDataBuffer);
        std::swap(d_encodedLifeData, d_encodedlifeDataBuffer);
      }
    }
  }
  else {
    for (size_t i = 0; i < iterationsCount; ++i) {
      bitLifeKernelLookup<<<blocksCount, threadsCount>>>(d_encodedLifeData,
        uint(worldEncDataWidth), uint(worldHeight), bytesPerThread, d_lookupTable,
        d_encodedlifeDataBuffer);
      std::swap(d_encodedLifeData, d_encodedlifeDataBuffer);
    }
  }
 
  checkCudaErrors(cudaDeviceSynchronize());
  return true;
}


void runSimpleLifeKernel(ubyte*& d_lifeData, ubyte*& d_lifeDataBuffer, size_t worldWidth,
  		size_t worldHeight, size_t iterationsCount, ushort threadsCount) {
  		assert((worldWidth * worldHeight) % threadsCount == 0);
  		size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
  		ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);
 
  		for (size_t i = 0; i < iterationsCount; ++i) {
    		simpleLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, worldWidth, worldHeight, d_lifeDataBuffer);
    		std::swap(d_lifeData, d_lifeDataBuffer);
  		}
}


__global__ void displayLifeKernel(const ubyte* lifeData, uint worldWidth, 
        uint worldHeight, uchar4* destination, int destWidth, int detHeight,
        int2 displacement, double zoomFactor, int multisample,
        bool simulateColors, bool cyclic, bool bitLife) {
 
    uint pixelId = blockIdx.x * blockDim.x + threadIdx.x;
 
    int x = (int)floor(((int)(pixelId % destWidth) - displacement.x) * zoomFactor);
    int y = (int)floor(((int)(pixelId / destWidth) - displacement.y) * zoomFactor);
 
    if(cyclic){
        x = ((x % (int)worldWidth) + worldWidth) % worldWidth;
        y = ((y % (int)worldHeight) + worldHeight) % worldHeight;
    }
    else if (x < 0 || y < 0 || x >= worldWidth || y >= worldHeight) {
        destination[pixelId].x = 127;
        destination[pixelId].y = 127;
        destination[pixelId].z = 127;
        return;
    }
 
    
    int value = 0;  // Start at value - 1.
    int increment = 255 / (multisample * multisample);
 
    if(bitLife){
        for(int dy = 0; dy < multisample; ++dy){
            int yAbs = (y + dy) * worldWidth;
            for(int dx = 0; dx < multisample; ++dx){
                int xBucket = yAbs + x + dx;
                value += ((lifeData[xBucket >> 3] >> (7 - (xBucket & 0x7))) & 0x1) * increment;
            }
        }
    }
    else {
        for (int dy = 0; dy < multisample; ++dy) {
            int yAbs = (y + dy) * worldWidth;
            for (int dx = 0; dx < multisample; ++dx) {
                value += lifeData[yAbs + (x + dx)] * increment;
            }
        }
    }
 
    
    bool isNotOnBoundary = !cyclic || !(x == 0 || y == 0);
 
    if(simulateColors) {
        if (value > 0) {
      if (destination[pixelId].w > 0) {
        // Stayed alive - get darker.
        if (destination[pixelId].y > 63) {
          if (isNotOnBoundary) {
            --destination[pixelId].x;
          }
          --destination[pixelId].y;
          --destination[pixelId].z;
        }
      }
      else {
        // Born - full white color.
        destination[pixelId].x = 255;
        destination[pixelId].y = 255;
        destination[pixelId].z = 255;
      }
    }
    else {
      if (destination[pixelId].w > 0) {
        // Died - dark green.
        if (isNotOnBoundary) {
          destination[pixelId].x = 0;
        }
        destination[pixelId].y = 128;
        destination[pixelId].z = 0;
      }
      else {
        // Stayed dead - get darker.
        if (destination[pixelId].y > 8) {
          if (isNotOnBoundary) {
          }
          destination[pixelId].y -= 8;
        }
      }
    }
    }
    else {
        destination[pixelId].x = isNotOnBoundary ? value : 255;
        destination[pixelId].y = value;
        destination[pixelId].z = value;
    }
    destination[pixelId].w = value;
}