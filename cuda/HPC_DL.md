# 面试问题整理

1. 基础的八股文：C/C++，OS，计算机体系结构等。这一部分略，网上已经有很多总结了。
2. 高性能计算基础知识：这一部分是面试的重点，本文章以CUDA为重点。
3. 各种AI框架知识：本文章以推理方向为主。
4. AI基础知识：对于常见的机器学习算法，以及CV & NLP & 推荐模型有一定了解，了解计算流程以及模型结构即可，重点为了能分析出计算瓶颈在哪里，找出可能优化的方向。本部分略
5. 算法题： 手写CUDA kernel和leetcode的比例大约为3:1。手写CUDA kernel的时候一般会结合第2部分一起问，一步一步要求你优化，每一步优化的具体原理，涉及到什么硬件知识等。

## 高性能计算基础

### 寒武纪芯片架构？

https://www.bilibili.com/video/BV1op4y157Qf  
https://www.bilibili.com/video/BV1op4y157Qf

### CUDA的线程组织结构

* warp是最小调度单元，一般有32个线程，对应的物理结构是 SMSP，根据不同结构，一个 SMSP可以驻留多个warp，warp之间切换无价，每个cycle可以发射一个准备好的warp，如果没有准备好的warp，则 stall。
* block，组织thread的单元，实际上被分成多个warp执行，一个block仅驻留在一个SM，ampere架构一个SM中有4个SMSP。
* grid组织多个block，分配block 到 SM驻留。CUDA的存储体系结构，

### 每一种存储的优缺点，该如何合理使用。

* register：最快的存储器，和计算单元同速，也存在bankconflect，代码层面无法避免，需要第三方工具微调机器码。一般用于存储线程中的局部变量。
* shared：block共享的高速可编程缓存。使用时也要注意bankconflect。另外也要注意大小，使用太多导致占用率下降，延迟掩藏效果变差。
* global：全局内存，主机可以通过API操作。设备可以直接访问，高延迟底带宽。使用时注意合并访存。
* local：位置同global，可见性权限为block，一般存储 1. 大的数组。2. 寄存器不够用
* const：位置同global，有高速cache，只读
* texture：位置同global，testure chache，访问空间相邻的数据有高速度，自动插值和范围检查

### GPU每一代的新特性有了解过吗？应该从哪里去了解详细信息？

* 官网查看 架构特性

### CUDA stream的概念，为什么要使用多个stream？

* 即发射的任务队列，一个 stream 里面的任务顺序执行。
* 多路并发管理，提高并发程度
* default stream 在所有其他stream结束后开始，其他stream在default stream结束后开始
* memcpyasync 要看有几个copyunit，而且要用pinned mem(cudamallochost)防止内存换页
* 三种同步： syncdevice syncstream waiteevent
* 隐式同步： 分配内存， memcpy，set

### GPU和CPU分别适合执行哪些程序？结合它们的硬件架构解释一下为什么它们有各自的优势。

#### CPU 低延迟：

* 通用计算 串行任务 控制流程 响应性任务（io）
* CPU 的 core 数量有限，并行程度低。虽然可以超线程但是 CPU 的线程是重实体，线程的上下文切换代价高（栈，寄存器，上下文）store & load

#### GPU 高吞吐：

* 并行计算任务 向量运算和矩阵计算 图形渲染和计算 深度学习和机器学习
* GPU 的线程最小执行单元是一个线程束(warp, 包含32个线程)，现代 GPU 设备每个 SM 最多 2048 个线程处于活动状态，GPU 可能有 80 SM 甚至更多。线程间切换几乎免费，如果一个线程需要等待，执行其他线程来覆盖等待时间就可以了

在现代计算中，常常使用 CPU 和 GPU 协同工作，以发挥它们各自的优势，提高整体计算性能。

### 说明一下神经网络加速器与CPU、GPU的区别，他们各自有何优势？

* CPU 仅有标量 ALU，现在 cpu 也有一些指令集支持 SIMD 指令
* GPU 可以加速神经网络应用和图像应用。有多个标量ALU，volta 引入 tensor core(更接近simd) 加速 GEMM 更好的支持神经网络计算。
* 加速器： SIMD (systolic arrays) 深度学习硬件加速器基于特定领域架构(Domain Specific Architecture， DSA)有更高的性能和更低的能耗。 一般都有标量，向量，矩阵的脉动阵列。

### 半精度浮点数FP16各个部分的具体位数，为什么要有半精度浮点数？

* 符号 - 指数 - 尾数  
float：1 - 8 - 32  
half：1 - 5 - 10

* 节省空间，提高计算速度，减小带宽压力，精度有微小下降
* 神经网络中，精度不需要很高即可满足推理需求，有些场景 int8 的精度即可满足需求了。

### TensorCore的加速原理
https://www.bilibili.com/video/BV1aL411a71w  
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#id42  
https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/  

* fma 需要 寄存器 - alu - 寄存器 - alu -寄存器 转移数据，tensorcore 不需要  
* 从 volta 开始引入，每个 SM 有 4 个 SMSP， 每个 SMSP 有 2 个tensor core，每个tensor core每个cycle可以提供 4x4x4的标量融合乘加，即 4x4 的矩阵乘加。  
* 每个SM即可提供 8 次这样的运算。  
* cuda编程中，WMMA api 打包了 16 x 16 矩阵乘加操作

```
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

### MPI，OpenMP以及CUDA各自适用的加速场景。

#### MPI（Message Passing Interface）：

MPI 是一种用于分布式内存并行计算的通信库，主要用于在多台计算机之间传递数据和协调计算。它适用于以下场景：

1. **大规模集群：** 当计算任务需要在多台计算机上进行协同计算时，MPI 可以用于实现节点之间的通信和数据传递。
2. **密集型计算：** 如果任务涉及大量的计算和通信，而每个节点上的计算任务相对独立，MPI 可以用于将任务分布到不同的计算节点上。
3. **科学计算：** MPI 在科学领域中常用于模拟、求解大规模方程组等需要高性能计算的应用。

#### OpenMP（Open Multi-Processing）：

https://www.bilibili.com/video/BV18M41187ZU  
OpenMP 是一种在共享内存系统中实现并行计算的技术，基于fork-join 模型，适用于：

1. **多核共享内存系统：** 在拥有多个处理核心的共享内存系统中，OpenMP 可以实现线程级别的并行计算，同时利用共享内存进行数据共享。
2. **循环并行：** OpenMP 适合并行化循环结构，将迭代计算任务分配给不同的线程执行。
3. **任务并行：** 对于一些可以分解为独立任务的应用，OpenMP 可以通过线程池执行这些任务并提高性能。

##### 编译:

g++ -fopenmp; cmake find_package(OpenMP) add_compile_options(-Wunknown-pragmas) target_link_libraries(xxx OpenMP::OpenMP_CXX)

##### 制导语句：

* #pragma omp \<directive name\> \<clause\> {}

###### parallel 构造

```
// {} 中的代码由多个线程并行执行
#pragma omp parallel xxx
{}:
```

###### for 构造

```
// 拆开for循环，分配给 parallel 构造得到的线程执行; 对for写法有要求
#pragma omp for xxx
for () {

}:
```

###### parallel for 构造

```
// parallel 和 for 指导语句经常合并
#pragma omp parallel for xxx
for () {

}
```

| 从句\制导语句                              | parallel | for | parallel for | 作用                                                                                          |
| ------------------------------------------ | -------- | --- | ------------ | --------------------------------------------------------------------------------------------- |
| if()                                       | V        |     | V            | 满足条件则并行                                                                                |
| num_threads()                              | V        |     | V            | 并行线程数量                                                                                  |
| default(shared\|none)                      | V        |     | V            | 指定默认变量类型，默认为shared                                                                |
| copyin                                     | V        |     | V            |                                                                                               |
| private(list)                              | V        | V   | V            | 每个线程生成一份主线程同名私有变量（未初始化）                                                |
| firstprivate(list)                         | V        | V   | V            | 同private，初始化为主线程的值                                                                 |
| shared(list)                               | V        | V   | V            | 共享变量列表，由编程人员保证共享变量的线程安全                                                |
| reduction(op : var)                        | V        | V   | V            | 给每个线程分配一个私有变量 var (初值和op相关)，出循环的时候按照 op 做 reduction               |
| lastprivate(list)                          |          | V   | V            | 同private，最后一个循环的私有数据复制给主线程的变量                                           |
| schedule(static\|dynamic\|guided\|runtime) |          | V   | V            | 如和将for分配给线程                                                                           |
| orderd                                     |          | V   | V            | 声明有潜在顺序部分，搭配 #pragma omp ordered 使用，ordered 区域内代码任意时刻最多一个线程执行 |
| collapse(n)                                |          | V   | V            |                                                                                               |
| nowait                                     |          | V   |              | 取消并行块结束的栅栏同步                                                                      |

###### sections 构造

```
#pragma omp sections
{
#pragma omp section
   code1();
#pragma omp section
   code2();
#pragma omp section
   code3();
}
```

###### barrier 构造

在特定位置栅栏同步 理解类似 cuda 的 __syncthreads() ?

```
// 这里是猜测的
#pragma omp sections
{
#pragma omp section
   code10();
#pragma omp barrier
   code11();
#pragma omp section
   code20();
#pragma omp barrier
   code21();
}
```

###### single 构造
###### master 构造
###### critical 构造
互斥的过程
###### atomic 构造
可用于 for 内部 原子操作
###### 任务 构造

允许定义任务以及依赖关系，动态调度执行
动态管理线程池和任务池

###### 向量化

自动优化：SIMD 使用AVX指令集

##### 内存墙 & 善用多级缓存

#### CUDA（Compute Unified Device Architecture）：

CUDA 是一种用于 GPU 计算的并行计算平台，适用于以下场景：

1. **并行向量化计算：** 当计算任务可以高效地进行向量化和矩阵计算时，CUDA 可以在 GPU 上执行高度并行的计算。
2. **深度学习和机器学习：** 训练和推断深度神经网络需要大量的矩阵计算，CUDA 在这些任务中表现出色。
3. **高性能计算：** CUDA 可以利用 GPU 强大的并行计算能力来加速科学计算、仿真、模拟等高性能计算任务。

总之，MPI 适用于分布式内存环境下的并行计算，OpenMP 适用于共享内存环境下的线程级并行，而 CUDA 则适用于在 GPU 上进行高度并行的向量化和矩阵计算。在选择适当的加速技术时，需要根据应用的性质和硬件配置来进行选择。

### RDMA相关问题。

DMA（Direct Memory Access，直接内存访问）是一种计算机系统中的技术，用于在不需要中央处理器（CPU）干预的情况下，实现外部设备和内存之间的数据传输。DMA 允许外部设备直接访问系统内存，从而减少了 CPU 参与数据传输的开销，提高了数据传输的效率和速度。

以下是 DMA 的一些关键特点和用途：

1. **减少 CPU 负担：** 在传统的数据传输过程中，CPU 需要在设备和内存之间进行数据拷贝和传输，导致 CPU 负担增加。DMA 技术允许外部设备直接和内存进行数据传输，减少了 CPU 的干预。
2. **高速数据传输：** 由于 DMA 可以直接访问内存，避免了 CPU 的中介，从而实现了高速的数据传输。这对于高带宽和低延迟的数据传输任务很有用。
3. **多通道支持：** 现代系统中常常具有多个 DMA 通道，允许多个设备同时进行数据传输，提高了系统的并行性能。
4. **外设和内存之间的数据交换：** DMA 可以用于设备之间的数据交换，如网络适配器将数据直接传输到内存中，或将数据从内存传输到磁盘中，而无需 CPU 的干预。
5. **实时数据传输：** 对于需要实时性能的任务，DMA 可以实现低延迟的数据传输，因为它避免了 CPU 的中介。
6. **大规模数据传输：** 对于大规模数据传输任务，如音视频流、大文件传输等，DMA 可以高效地处理大量数据。

总的来说，DMA 技术通过直接内存访问，实现了外部设备和内存之间的高效数据传输，减少了 CPU 的负担，并提高了系统的性能。它在高速数据传输、实时性能要求和大规模数据传输等场景中具有广泛的应用。

RDMA（Remote Direct Memory Access）是一种高性能计算和通信技术，用于在分布式计算环境中实现直接内存访问。RDMA 允许一台计算机直接从另一台计算机的内存中读取或写入数据，而不需要涉及中间的 CPU 或操作系统参与。

以下是 RDMA 的一些关键特点和用途：

1. **零拷贝传输：** RDMA 可以在两台计算机之间实现数据传输，而不需要在发送和接收数据时进行额外的内存拷贝，从而减少了数据传输的开销。
2. **低延迟：** 由于 RDMA 可以直接访问内存，避免了操作系统和网络协议的干预，因此可以实现低延迟的数据传输，适用于实时性要求高的应用。
3. **高带宽：** RDMA 可以充分利用网络带宽，实现高速数据传输，适用于大规模数据传输和计算密集型任务。
4. **并行计算：** RDMA 可以在分布式计算集群中支持并行计算，允许多台计算机之间进行高效的数据共享和通信。
5. **高性能计算：** RDMA 在科学计算、大数据分析和高性能计算领域具有广泛的应用，可以加速数据传输和计算任务。
6. **网络协议：** RDMA 可以在不同的网络协议上实现，例如 InfiniBand、RoCE（RDMA over Converged Ethernet）等。
7. **数据中心应用：** RDMA 技术也在数据中心中得到广泛应用，用于加速分布式存储、虚拟化、容器化等场景。

总的来说，RDMA 技术通过直接内存访问和高性能的数据传输，提供了一种在分布式计算环境中高效通信和数据共享的方式。它在高性能计算、数据中心、大规模集群等场景中发挥着重要作用。

### 如何进行kernel的优化，会用到哪些工具？

#### 工具：

cnperf, nsight compute

### kernel 的优化点和一些优化技巧：

合并访存，shared_mem，solve bank conflict，原子操作，warp_shuffle  

#### 向量化LD
https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/  
reduces the total number of instructions, reduces latency, and improves bandwidth utilization.  

#### 网格跨步循环
https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/  

#### 异步数据传输（global -> shared）数据预取 double buffer
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#with-memcpy-async  
https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/

#### 高效的库：

libcuxx, thrust, cutlass, CUB, mordengpu

##### cutlass
将矩阵乘法的过程分为读，写，块计算等的组件，用户根据自己的场景使用这些组建实现自己的高性能矩阵乘法计算  
也可以自定义的在矩阵乘epilogue融合其他操作达到自定义的算子融合
head only , 允许编译器优化  
https://github.com/NVIDIA/cutlass/issues/109  
https://www.bilibili.com/video/BV13w411o7cu


### 如何进行system的优化，会用到哪些工具？

低延迟还是高吞吐？

nsight system  
无依赖算子是否最大程度并发  
launch kernel 开销是否是瓶颈  
异步多流拷贝是否最大化利用了全部DMA  
是否还有可融合的算子  
前后处理是否是瓶颈

### CPU上有哪些并行优化方法？

1. **多线程编程（OpenMP）：** 利用多线程技术，将任务分解成多个线程，让多个线程在不同的处理核心上同时执行。这可以提高 CPU 的利用率和计算效率。
2. **向量化 SIMD（Single Instruction, Multiple Data）指令：** SIMD 是一种在单个指令中执行多个数据操作的技术。通过向量化编程，可以使 CPU 在同一时钟周期内处理多个数据，提高计算吞吐量。
3. **NUMA（Non-Uniform Memory Access）优化：** 针对 NUMA 架构的多核 CPU，优化内存访问，使得每个核心能够更快地访问本地内存，从而提高访问效率。
4. **数据重排：** 优化数据布局，使得数据在内存中连续存放，减少缓存失效和内存访问开销。
5. **并行算法设计：** 设计适合并行计算的算法，将计算任务分解为可并行执行的部分。
6. **CPU Affinity：** 控制线程在特定的 CPU 核心上运行，以避免频繁的核心切换，提高缓存命中率。

### ARM相关的库有了解过吗？

### PTX有了解过吗？

Parallel Thread eXecution，PTX

1. **中间语言：** PTX 是一种中间语言，它提供了对 GPU 硬件特性的抽象，使开发者能够更容易地进行 GPU 编程。
2. **平台独立：** PTX 是与 GPU 架构无关的，这意味着同一份 PTX 代码可以在不同的 NVIDIA GPU 架构上执行，而不需要针对每种架构编写不同的代码。
3. **可移植性：** PTX 提供了更高的代码可移植性，开发者可以通过编写 PTX 代码，使其在不同的 GPU 架构上获得最佳性能。
4. **优化：** PTX 允许开发者更精细地控制 GPU 上的指令流程，以实现更高效的计算。开发者可以在 PTX 级别进行优化，以充分利用底层硬件的特性。
5. **汇编代码：** PTX 可以看作是一种高级汇编代码，但它不同于特定的 CPU 汇编，而是针对 GPU 设计的。
6. **CUDA 编译器：** CUDA 编译器将高级的 CUDA C/C++ 代码编译成 PTX 中间代码。然后，由 CUDA 驱动程序将 PTX 进行最终的编译，生成可在 GPU 上执行的机器代码。

### roofline模型有什么用？

https://www.codee.com/is-your-algorithm-running-at-peak-performance-the-roofline-model/

* flops = flop/byte * byte/s = operational intensity * bandwidth  
* log(FLOPS) = log(AI) + log(BW)  
* 点的位置可以提示可以做那方面的优化。  
* 实际带宽（点的斜率）低于峰值带宽是因为并不是一直在读数据，可以空过数据与预取，smem，提高cache命中率，访存对齐，改善性能。  
* 计算强度在拐点左侧可以通过提升计算强度提升性能。  

### 如何确定最优的BLOCK_SIZE

1. 高的占用率更有可能有好的延迟掩藏，合理的使用shared_mem register资源，适当提高占用率。
2. 减少数据padding导致的计算资源的浪费

* 在选择block大小时，至少保证每个SM都能分到block。
* 应该使用至少SMSP个warp的块，但只有在每个多处理器上有多个并发块时。
* 每个块的线程数应该是线程束大小的倍数，以避免在线程束中浪费计算资源，并促进合并操作。
* 如果延迟影响性能，应该使用多个较小的线程块而不是一个较大的线程块来驻留在每个多处理器上。这对于经常调用__syncthreads()的内核尤其有益。

* 在每个块中选择128到256个线程是进行不同块大小实验的良好起始范围。

### GPU资源调度有哪些方法？
k8s, 虚拟化？

### 稀疏矩阵的存储格式有哪些？稀疏矩阵的应用场景？稀疏矩阵计算与稠密矩阵计算有何不同？

**压缩稀疏行（Compressed Sparse Row，CSR）**

**压缩稀疏列（Compressed Sparse Column，CSC）**

**坐标（Coordinate，COO）**

**对角线（Diagonal）**

1. **图和网络分析：** 社交网络、网络图、推荐系统等领域中的数据通常可以表示为稀疏矩阵，其中矩阵的行和列表示节点，矩阵的元素表示节点之间的关系。
2. **自然语言处理（NLP）：** 在自然语言处理中，词袋模型、文本分类、文本相似度等任务中，词汇表往往很大，但是每个文档中只包含一小部分词汇，从而导致稀疏矩阵的形式。
3. **计算流体力学（CFD）：** 在计算流体力学中，矩阵通常表示物理现象的离散化模型。对于大规模问题，矩阵的非零元素很少，因此使用稀疏矩阵存储可以减少内存占用。
4. **有限元分析（FEA）：** 类似于CFD，有限元分析中的离散化模型可以表示为稀疏矩阵，例如在结构分析、热传导分析等领域。
5. **图像处理：** 在图像处理中，图像可以被分解成稀疏的频域表示，例如使用稀疏变换（如小波变换）。
6. **推荐系统：** 推荐系统中的用户-物品评分矩阵通常是稀疏的，因为用户只对少数物品进行了评分。
7. **机器学习：** 在某些机器学习算法中，特征之间的关联性不高，导致特征矩阵是稀疏的，例如文本分类中的词频矩阵。
8. **信号处理：** 在信号处理中，信号可以表示为频域中的稀疏表示，例如在压缩感知中使用稀疏矩阵。
9. **地理信息系统（GIS）：** 地图数据和地理信息可以表示为稀疏矩阵，其中每个点可能对应一个地理坐标和属性。
10. **数据结构：**
    * 稀疏矩阵：在稀疏矩阵中，大多数元素为零。因此，不同的稀疏矩阵存储格式（如CSR、CSC、COO等）被设计用来有效地存储非零元素，从而减少内存占用。
    * 稠密矩阵：稠密矩阵中的每个元素都有值，因此可以使用连续的内存块存储整个矩阵，不需要特殊的存储格式。
11. **计算效率：**
    * 稀疏矩阵：由于稀疏矩阵中大部分元素是零，对于稀疏矩阵的计算，只需考虑非零元素，从而减少计算量。此外，稀疏矩阵的存储格式也能够在计算时提高缓存利用率，减少内存访问开销。
    * 稠密矩阵：稠密矩阵的计算通常涉及所有元素，因此需要更多的计算资源和内存带宽。
12. **计算复杂性：**
    * 稀疏矩阵：由于稀疏矩阵的计算只涉及非零元素，一些操作（如矩阵-向量乘法、矩阵乘法等）的计算复杂性可能随稀疏度的降低而减少。
    * 稠密矩阵：稠密矩阵的计算涉及所有元素，因此在一些操作中，计算复杂性可能更高。
13. **内存占用：**
    * 稀疏矩阵：由于稀疏矩阵存储了较少的非零元素，其内存占用要比稠密矩阵小得多，特别是在大规模问题上。
    * 稠密矩阵：稠密矩阵存储了所有元素，其内存占用通常较大，可能会限制问题规模。

### 如何计算CPU指令的吞吐量和时延?

## AI 框架知识

这一部分会涉及一些AI框架(训练&推理&编译器)相关的问题，并且会重点根据简历上的项目经历去做一些发散性的提问。

### MLIR有了解过吗？ONNX有了解过吗？

1. **什么是 MLIR？**

   * MLIR（Multi-Level Intermediate Representation）是一个开源的领域特定语言（DSL）和编译器基础设施，旨在帮助优化和转换各种编程语言的中间表示。MLIR 的设计目标是为不同领域和应用提供一个统一的中间表示，以支持高效的编译优化、代码生成和代码转换。

   1. **多级别表示：** MLIR 提供多级别的中间表示，这些表示可以用于不同的编译优化和转换阶段。它可以用于高级的领域特定语言（DSL），也可以映射到低级别的表示，以便进行更底层的优化和生成。
   2. **领域特定语言支持：** MLIR 支持通过定义领域特定语言（DSL）来描述特定领域的语义和操作。这使得在特定领域中进行优化和转换更加方便。
   3. **模块化和可扩展性：** MLIR 的设计使得添加新的中间表示和优化传递变得更加容易。它提供了灵活的模块化架构，可以根据需要添加新的优化和转换。
   4. **代码生成和优化：** MLIR 的目标之一是支持高效的代码生成和优化。通过中间表示的层次结构，可以在不同层级上应用各种优化策略。
   5. **开放社区：** MLIR 是一个开源项目，受到了编译器、领域特定语言、机器学习等领域的关注。它的设计和发展受到了来自多个组织和社区的贡献。
2. **MLIR 如何支持多级别中间表示？**

   1. **高层表示（High-Level Representation）：**
      * 高层表示通常是针对特定领域的领域特定语言（DSL）。每个 DSL 针对特定的应用领域，具有特定的语义和操作。
      * 高层表示使开发人员能够使用更接近问题领域的语法和概念来表达代码逻辑。
   2. **中层表示（Intermediate-Level Representation）：**
      * 中层表示是一个通用的、可扩展的中间形式。它是连接不同领域和不同硬件平台的桥梁。
      * 高层表示可以通过映射和转换机制转换为中层表示。中层表示可以进行多种优化和转换，不受特定领域的限制。
   3. **低层表示（Low-Level Representation）：**
      * 低层表示通常与底层硬件平台相关。它可以是与特定硬件架构对应的表示，用于代码生成和最终执行。
      * 中层表示可以映射到低层表示，以生成最终的机器码或硬件描述语言。

   不同级别的中间表示之间的关系如下：

   * **高层到中层映射：** 高层表示可以通过映射和转换机制转换为中层表示。这个过程可以包括将高层语义映射到中层操作，以及进行一些高级优化。
   * **中层到低层映射：** 中层表示可以映射到底层硬件描述，这通常涉及将通用的中层操作映射到特定硬件指令。
   * **反向转换和优化：** 从低层回到中层，以及从中层回到高层的转换也是可能的。这可以用于进行后端优化、性能分析以及在不同级别之间的转换。

### TVM的整体结构，如何用TVM进行开发？

TVM（Tensor Virtual Machine）是一个用于优化和部署深度学习模型的开源框架，它支持跨多种硬件平台进行高效的模型编译和执行。TVM 的整体结构包括以下主要组件：

1. **Frontend：** Frontend 负责将深度学习框架（如PyTorch、TensorFlow）中的模型导入到 TVM 中。这些模型通常以图形计算图（Graph）的形式表示。
2. **中间表示（Intermediate Representation，IR）：** TVM 使用 Relay 作为其中间表示，它是一种面向图形计算图的高级中间表示。Relay 提供了丰富的操作和优化策略，用于在不同硬件上进行模型优化和转换。
3. **优化器（Optimizer）：** 优化器在 Relay 中进行各种优化，包括常量折叠、图优化、算子融合等。这些优化可以在模型执行前进一步提升模型性能。
4. **后端（Backend）：** 后端负责将 Relay 中的优化后的模型转换为特定硬件的代码。TVM 支持多种后端，如 LLVM、CUDA、OpenCL 等，以及专门针对不同硬件的后端。
5. **编译器（Compiler）：** 编译器将 Relay 中的优化模型编译为目标后端所支持的代码。TVM 在这个阶段会进行低级别的优化，生成高效的硬件代码。
6. **运行时（Runtime）：** 运行时负责执行编译后的模型。它包括了与硬件平台交互的组件，使得模型能够在特定硬件上高效运行。

在使用 TVM 进行开发时，一般的流程如下：

1. **定义模型：** 使用你熟悉的深度学习框架定义模型，如 PyTorch 或 TensorFlow。
2. **导入模型：** 使用 TVM 的前端将模型导入到 TVM 中，通常以 Relay 图的形式。
3. **优化模型：** 使用 TVM 的优化器对模型进行各种优化，以提高性能。
4. **选择后端：** 选择适合的后端（如 CUDA、OpenCL）来将模型编译为硬件代码。
5. **编译模型：** 使用 TVM 的编译器将优化后的模型编译为目标后端的代码。
6. **执行模型：** 使用 TVM 的运行时来执行编译后的模型，使其在特定硬件上运行。
7. **调优和部署：** 在实际硬件上测试和调优模型，根据需求对模型进行部署。

TVM 提供了一系列 API 和工具，使开发者能够在不同硬件上高效地优化、编译和执行深度学习模型。通过熟悉 TVM 的整体结构和组件，你可以更好地利用 TVM 进行模型的开发和部署。

### 为什么要进行推理优化？直接用tensorflow或者pytorch的推理接口不行吗？

AI框架主要是为算法科学家设计的，这个设计定位使得AI框架在设计取舍上会倾向于灵活性，即使得算法科学家能够用框架搭出尽可能多的新模型架构。这就要求在算子设计颗粒度上要尽可能小。  
而在生产环境中，主诉求从灵活变成了效率，因为成本很重要。而效率要求做一些激进的优化  
如做一些大的operator fusion，数学变换以及微架构友好的优化， 从而减少调度开销和微架构开销。可以看到生产环境和研究环境下对框架设计的要求是在“灵活和效率”两大矛盾的两端，这种矛盾是很难调和的。基于以上，在这种情况下，为生产环境专门做一个效率为先的框架是必要的，这一块也有越来越多的工程师的投入。

### 模型推理优化的常用方法有哪些？

#### **算子优化**

算子优化的题中之意就是优化单算子的性能，方法无非是算法优化和微架构优化。

* 算法优化。对同一个算子可能有不同的算法去实现它。举个例子，对卷积算法我们常见的就有：矩阵乘法，直接卷积法，Winograd变换法，FFT变换法。需要我们针对目标问题，分析并实现合适的算法，来达到最佳性能。
* 有效带宽 和 访存计算比 优化
* 微架构优化。微架构优化主要焦点是如何充分利用好微架构的内置加速器的能力去最大化算子的性能。tensor core，sfl，vectorized-memory-access 等特色功能。

#### **图优化**

图优化主要通过子图变换和算子融合的方式来达到减少计算量或者其他系统开销（如访存开销），从而达到性能优化的目的。图优化主要是希望在不影响模型的数值特性的基础上，通过图变换达到简化计算、资源开销，提升性能，所以是性能优化时的首选方法之一。下面列举几例。

* **子图变换**
  子图变换主要是通过数学变换的手段对图进行精简，从而减少计算和调度开销。常见的有常数折叠，公共子表达式折叠(common subexpression elimination (CSE) )以及算术变换。
* **算子融合 (Operator Fusion/Remapper)**
  算子融合基于对深度学习拓扑结构模式的观察。深度学习算子按其对资源的需求可以分为两类：

  * 计算密集型算子，这些算子的时间绝大部分花在计算上，如卷积、全连接等。
  * 访存密集型算子，这些算子的时间绝大部分花在访存上，他们大部分是element-wise算子，active，transpose。

  在典型的深度学习模型中，一般计算密集型和访存密集型算子是相伴出现的，最简单的例子是”Conv + ReLU“相伴出现。这时候我们可以通过fusion来实现“in register computing”，从而减少访存密集型算子的访存，减少内存访问延时和带宽压力，提高推理效率。**算子的衔接是globalmem读写，很慢的。而且可以省去layout转换操作**

#### **模型压缩**

  上面的都是无损的，当这三点都做完了后，如果还需要额外的性能增益，这时候需要考虑模型压缩方案。模型压缩(Compression)主要手段有：模型量化、模型蒸馏和模型稀疏化。

* 模型量化。模型量化主要是通过降低模型中tensor和weights精度的手段，从而减少计算需求和[数据存储](https://cloud.tencent.com/product/cdcs?from_column=20065&from=20065)与传输需求，来达到加速的目的。主要方法分两派：一是训练后量化(Post-training Quantization)，二是量化感知训练(Quantization-Aware Training)。这个topic比较大，可以另讲。
* 模型蒸馏。模型蒸馏采用的是迁移学习的方法，通过采用预先训练好的复杂模型(Teacher Model)的输出作为监督信号去训练另外一个简单的网络(Student Model)，最后把Student Model用于推理。
* 模型稀疏化。稀疏化首先是Han Song做FPGA的时候提出来的，这个时期的稀疏化就是纯稀疏化，减少synapses。这种稀疏化，适合FPGA，相当于减少了电路。但对于通用计算平台（如CPU，[GPU](https://cloud.tencent.com/product/gpu?from_column=20065&from=20065)等），不规则稀疏矩阵计算最终还是要落地到规则的计算单元上，这个时候是性能变好还是变差取决于problem size和如何把问题映射到计算单元上的方案，性能是提高还是降低是不确定的。所以后面业界的研究重点就转向了结构稀疏化

#### **部署优化**

##### 推理形式
* 按照拓扑排序的顺序串行 launch kernel
* 根据层的依赖关系，没有依赖的 kernel launch 在不同 stream


  前面三类优化可以认为是静态优化，是与资源分配无关的。但当这些静态优化部署在平台上的时候，就产生了executable和部署环境的交互，这就产生了资源分配和调度的问题。部署优化主要通过调整模型在部署时的资源分配和调度的参数来进一步优化性能。

### 算子融合为什么能加速推理，优化了哪一部分？TensorRT用到了哪些算子融合？算子融合在推理框架中是如何实现的？

https://openmlsys.github.io/chapter_backend_and_runtime/graph_optimizer.html

计算密集和带宽密集算子融合，更好的利用带宽和计算资源。

省去了launchkernel的成本，

节省数据写回globalmem再加载回寄存器计算的开销

有些算子需要重排数据layout，计算完成再还原，之后下一个算子可能又要重排。

---

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#fusion-types

---

1. 遍历路线：针对特定设备，枚举实现典型的CB+MB形式的融合算子，如Conv+ReLU/Conv+BN+ReLU/Dense+ReLU/Conv+Sum等等，Intel的oneDNN以及国内很多大厂的推理框架走的都是这个路线。
2. 规则路线：基于规则实现算子融合，以TVM为例，其将所有算子分为Opaque/Injective/Reduction/Complex-out四类，并设定融合规则形如Complex-out+Injective/Injective+Injective/Injective+Reduction等，以此实现所有符合规则的子图融合。
3. apollo 多层规约
   https://openmlsys.github.io/chapter_backend_and_runtime/graph_optimizer.html
   https://zhuanlan.zhihu.com/p/578274625

   首先是图划分阶段，提取DNN模型的所有subgraph，并将subgraphs根据预定的规则重新分解并聚合成更合理的micrograph，这一步的目的是让所有可能后续做融合的子图聚合为micrograph，为后续所有融合工作划定范围
   再就是基于以上拆分得到的micrographs做了bottom-up的重新融合，分为三步：

   a. 基于Polyhedral启发算法做loop fusion，这一阶段将诸如Dense+ReLU、Conv+ReLU形式的子图做循环级别的融合
   b. 继续融合优化非CB算子，此步骤参考了阿里的AStitch工作，通过对计算图算子级别依赖和元素级别依赖的分析，将MB算子尽可能融合
   c. 基于以上的融合结果，做了类似TASO和RAMMER的工作，识别无计算依赖的算子做并行计算


### 有研究过某一个框架的具体源码吗？

### TensorRT如何进行自定义算子开发？

1. https://oldpan.me/archives/tensorrt-plugin-one-post-get-it

### TensorRT对模型实现了哪些推理优化？

https://zhuanlan.zhihu.com/p/432215219

模型量化

算子融合

kernel 自动调优

多流执行

动态内存


### 模型量化的加速原理，模型量化带来的精度损失如何解决？

降低带宽压力，降低内存消耗，降低能耗

量化的数据类型有专用计算单元，移动端出于功耗考虑可能值支持int8 fp16的计算

---

1. 非对称量化：对称量化将量化范围对称地分布在零点周围，而非对称量化允许量化范围在零点两侧不对称分布。非对称量化通常能够提供更好的精度，因为它可以更好地适应数据的分布。
2. 非线性量化：实际的深度神经网络的权重和激活值通常是不均匀的，因此理论上使用非线性量化导致的精度损失更小，但在实际推理中非线性量化的计算复杂度较高
3. 量化粒度：per_tensor,perchannel,per_row
4. 感知量化训练：在模型训练过程中加入伪量化算子，通过训练时统计输入输出的数据范围可以提升量化后模型的精度，适用于对模型精度要求较高的场景
5. 校准：获取 activation 的 scale 和 zero point 数据
6. 混合精度：对不同的层按精度需求考虑是否量化

---

训练后量化：权重量化和全量化，权重量化仅量化模型的权重以压缩模型的大小，在推理时将权重反量化为原始的float32数据，后续推理流程与普通的float32模型一致。权重量化的好处是不需要校准数据集，不需要实现量化算子，且模型的精度误差较小，由于实际推理使用的仍然是float32算子，所以推理性能不会提高。全量化不仅会量化模型的权重，还会量化模型的激活值，在模型推理时执行量化算子来加快模型的推理速度。为了量化激活值，需要用户提供一定数量的校准数据集用于统计每一层激活值的分布，并对量化后的算子做校准。校准数据集可以来自训练数据集或者真实场景的输入数据，需要数量通常非常小。在量化激活值时会以校准数据集为输入，执行推理流程然后统计每层激活值的数据分布并得到相应的量化参数

### ONNX Runtime支持在多种硬件上进行推理，说明具体的实现机制。

开放神经网络交换（Open Neural Network Exchange）简称ONNX是微软和Facebook提出用来表示深度学习模型的**开放**格式。所谓开放就是ONNX定义了一组和环境，平台均无关的标准格式，来增强各种AI模型的可交互性。

ONNX Runtime 是一个开源的深度学习推理引擎，支持在多种硬件上进行推理，包括 CPU、GPU、FPGA 等。ONNX Runtime 实现了针对不同硬件的后端（Backend），每个后端都有自己的实现机制以支持推理。

以下是 ONNX Runtime 在不同硬件上进行推理的一般实现机制：

1. **CPU 后端：**
   * ONNX Runtime 的 CPU 后端主要使用基于线程的并行执行来加速推理。它利用多线程在 CPU 上同时执行操作，充分发挥 CPU 的多核能力。
   * CPU 后端还利用了SIMD（Single Instruction, Multiple Data）指令集，对于支持SIMD的指令集（如AVX、AVX2、AVX512等），可以将多个数据同时传递给同一指令，实现高效的并行计算。
2. **GPU 后端：**
   * GPU 后端使用 CUDA 或 OpenCL 来实现对 GPU 的并行计算。它可以将模型操作映射到 GPU 上的线程块和线程，并利用 GPU 的并行计算能力加速推理。
   * GPU 后端还会利用 GPU 的硬件特性，如向量化指令和高速内存，来提高计算效率。
3. **FPGA 后端：**
   * FPGA 后端利用 FPGA（Field-Programmable Gate Array）的可编程硬件特性，将模型操作映射到 FPGA 上进行定制化的硬件加速。
   * FPGA 后端通常需要针对具体的 FPGA 平台进行优化和定制，以获得最佳的性能。
4. **其他硬件后端：**
   * ONNX Runtime 还可以支持其他硬件后端，如边缘设备上的专用加速器。这些后端的实现方式会根据硬件的特点和能力而有所不同。

总体而言，ONNX Runtime 通过为不同硬件平台实现特定的后端，利用硬件的并行计算能力、向量化指令等特性，以及各种优化技术，实现了跨多种硬件平台的高效推理。这种模块化的后端设计使 ONNX Runtime 能够适应不同硬件的性能特点，提供高性能的推理解决方案。

### 总结一下TensorRT，ONNX Runtime等推理框架的组成架构，如果我们公司自己要为硬件开发一套推理框架，应该重点关注哪些部分？

先明确需求，为什么开发推理引擎，根据需求确定功能。

1. **算子支持和优化：** 实现常见的深度学习算子，如卷积、池化、全连接等，优化计算流程以提高性能。考虑支持算子融合、自动调优等方法。
2. **多平台支持：** 考虑框架能够在多种硬件平台上执行，如 CPU、GPU、FPGA、边缘设备等。为每种平台进行适当的硬件加速和优化。
3. **性能优化：** 通过量化、混合精度计算、常量折叠等方法提高推理性能。考虑编译技术、算子融合等优化。
4. **模型转换和兼容性：** 支持从主流深度学习框架（如 TensorFlow、PyTorch）导入模型，并确保模型在框架中能够正确执行。考虑支持 ONNX 格式等中间表示。
5. **自动硬件适配：** 考虑框架能够自动适配不同硬件平台的特性和限制，实现自动化硬件加速。
6. **量化和精度控制：** 考虑支持模型量化以减少内存占用和计算开销，同时提供精度控制以满足不同应用场景的需求。
7. **多线程和并发：** 支持多线程并发执行，充分利用多核 CPU 和 GPU 的并行计算能力。
8. **错误处理和调试：** 提供清晰的错误处理和调试信息，帮助开发人员定位和解决问题。
9. **性能分析工具：** 提供性能分析工具，帮助开发人员识别性能瓶颈和优化机会。
10. **文档和示例：** 提供详细的文档和示例代码，帮助开发人员快速上手并理解如何使用框架。
11. **社区支持：** 创建一个积极的社区，为用户提供技术支持、问题解答和新特性的建议。
12. **安全性和隐私：** 考虑模型推理过程中的数据隐私和安全问题，提供相应的机制和功能。
13. **灵活性和可扩展性：** 考虑支持不同模型结构和扩展性需求，允许开发人员自定义算子、后端等。
14. **与其他框架的互操作性：** 考虑与其他深度学习框架的互操作性，允许模型在不同框架之间无缝切换和执行。
15. **维护和更新：** 持续维护和更新框架，跟踪深度学习领域的最新发展，提供新功能和优化。

### 各种推理框架都有何优劣势？它们的性能怎么样？

不同的深度学习推理框架各有优劣势，它们在性能、功能、适用场景等方面有所不同。以下是一些常见的深度学习推理框架以及它们的优劣势：

1. **TensorFlow Lite:**
   * 优势：适用于移动设备、嵌入式设备和边缘设备，支持模型量化、硬件加速和低功耗特性。
   * 劣势：可能在某些情况下相对于其他框架的性能略低，不太适合大规模的服务器端部署。
2. **ONNX Runtime:**
   * 优势：支持多种硬件和平台，有广泛的硬件后端支持，具有较高的性能和灵活性。
   * 劣势：在某些情况下可能对某些硬件不够优化，可定制性相对较弱。
3. **OpenVINO:**
   * 优势：适用于英特尔 CPU、GPU、FPGA 等硬件，具有较好的硬件支持和性能。
   * 劣势：在非英特尔硬件上可能不太适用，不太灵活，适用范围受限。
4. **NVIDIA TensorRT:**
   * 优势：适用于 NVIDIA GPU，具有非常高的性能和优化效果，支持混合精度计算、算子融合等优化技术。
   * 劣势：仅适用于 NVIDIA GPU，不适用于其他硬件。
5. **TVM:**
   * 优势：具有编译和优化功能，支持多种硬件和后端，提供了较高的自由度和可定制性。
   * 劣势：配置和优化可能需要一定的技术和工作量，不太适合初学者。
6. **Core ML:**
   * 优势：适用于 iOS 和 macOS 设备，与苹果生态良好集成，性能较高。
   * 劣势：只能在苹果设备上使用，不适用于其他平台。
7. **SNPE:**
   * 优势：适用于 Qualcomm Snapdragon 平台，具有硬件优化，性能较好。
   * 劣势：仅适用于特定的 Qualcomm 平台，不适用于其他硬件。
8. **MNN:**
   * 优势：适用于移动设备、嵌入式设备，轻量级，具有较好的性能。
   * 劣势：定制性可能相对较弱，适用范围有限。

### 分布式训练中有哪些并行模式？每种模式需要做什么，有什么优缺点？

在分布式训练中，存在多种并行模式，用于加速训练过程并提高训练效率。以下是常见的几种并行模式及其特点、优缺点：

1. **数据并行（Data Parallelism）：
   解决算力不足问题**

   * **异步并行（Asynchronous Parallelism）：**

     * **模式：** 在异步并行模式中，各个设备独立地计算梯度并更新参数，而不等待其他设备的梯度计算完成。
     * **优点：** 提高设备的利用率，可以避免等待通信导致的训练效率下降。
     * **缺点：** 异步更新可能会导致不稳定的训练过程，参数更新可能存在冲突。
   * **同步并行（Synchronous Parallelism）：**

     * **模式：** 在同步并行模式中，各个设备会等待所有设备的梯度计算完成后，进行参数更新，确保参数同步。
     * **优点：** 训练过程相对稳定，参数同步保证了参数一致性。
     * **缺点：** 可能会因为等待通信导致效率下降，尤其是在设备之间通信延迟较大时。
2. **模型并行（Model Parallelism）：
   解决内存不足问题**

   * **模式：** 在模型并行模式中，模型的不同部分在不同的设备上并行处理，通常用于较大的模型，其中的参数可以分布在多个设备上。
   * **优点：** 可以处理大型模型，提高模型容量，允许训练更大规模的模型。
   * **缺点：** 参数同步和通信可能是一个挑战，需要细致的模型拆分和调度。
3. **混合并行（Hybrid Parallelism）：**

   * **模式：** 混合并行模式结合了数据并行和模型并行，可以在多个维度上进行并行处理，从而充分利用多设备和多参数的优势。
   * **优点：** 结合了数据并行和模型并行的优势，适用于复杂的大型模型和多设备情况。
   * **缺点：** 复杂性较高，需要综合考虑参数同步、通信开销等因素。

### 分布式训练中我们重点需要处理的问题有哪些？目前已有哪些解决方案

在分布式训练中，需要处理一些关键问题以确保训练的有效性、性能和可靠性。以下是一些重点需要处理的问题以及目前已有的一些解决方案：

1. **数据分发和加载：** 将训练数据有效地分发到不同的设备上，以便并行训练。
   * **解决方案：** 数据并行、分布式数据存储系统（如HDFS、Ceph）、数据预处理和加载优化。
2. **参数同步：** 确保不同设备上的参数保持一致，以避免模型发散或不稳定。
   * **解决方案：** 同步梯度平均、同步参数更新、参数服务器等方法。
3. **通信开销：** 分布式训练会引入通信开销，影响训练速度。
   * **解决方案：** 优化网络通信，减少参数同步频率、异步更新、压缩通信等。
4. **调度和分配：** 合理分配计算资源，避免设备闲置或过载。
   * **解决方案：** 动态资源分配、设备亲和性、任务队列和调度器等。
5. **故障恢复：** 处理设备故障、通信中断等问题，保证训练的稳定性。
   * **解决方案：** 容错机制、检查点和恢复、备份设备等。
6. **异构设备：** 处理不同类型的硬件设备，如GPU、TPU、FPGA等。
   * **解决方案：** 支持多种硬件后端、硬件特定的优化、设备亲和性。
7. **模型切分和合并：** 将模型划分到不同的设备上，进行分布式计算。
   * **解决方案：** 模型并行、分布式模型切分、模型合并和同步。
8. **批量大小和学习率：** 在分布式训练中，批量大小和学习率可能需要调整。
   * **解决方案：** 批量大小缩放、学习率缩放、自适应学习率等。
9. **超参数调优：** 在分布式训练中，超参数的选择可能影响训练性能。
   * **解决方案：** 分布式超参数搜索、自动调参工具。
10. **性能监控和调优：** 监控训练性能，发现和解决性能瓶颈。
    * **解决方案：** 性能分析工具、监控和日志分析。

目前已有的解决方案包括 TensorFlow的分布式训练、PyTorch的分布式包、Horovod等框架，以及一些自研的分布式训练方案。不同的框架和工具提供了不同程度的支持，可以根据具体需求选择合适的解决方案。

### MPI如何应用于AI框架中？

在分布式深度学习训练中，有一些框架使用了MPI（Message Passing Interface）来实现进程间通信和协同工作。以下是一些使用了MPI的深度学习框架的示例：

1. **TensorFlow：** TensorFlow支持多种分布式训练策略，其中一些使用了MPI作为底层通信协议。例如，TensorFlow的分布式训练中的参数服务器策略和Horovod框架使用了MPI来实现进程间的通信。
2. **PyTorch：** PyTorch提供了 `torch.distributed`模块，用于实现分布式训练。该模块支持使用不同的后端，包括MPI、NCCL、Gloo等。因此，可以通过配置选择使用MPI作为通信协议。
3. **Horovod：** Horovod是一个Uber开源的分布式训练框架，它专门用于TensorFlow和PyTorch。Horovod使用了MPI来实现分布式训练中的通信和同步操作。
4. **ChainerMN：** ChainerMN是Chainer的一个分布式训练扩展库，使用了MPI来实现分布式训练中的通信和协同工作。
5. **Caffe2：** Caffe2也支持分布式训练，其中一些策略使用了MPI来进行进程间通信。

需要注意的是，并不是所有的深度学习框架都使用了MPI，一些框架可能使用其他的通信协议和库来实现分布式训练，例如NCCL、Gloo等。选择使用特定的分布式框架和通信协议取决于具体的需求、硬件环境和框架的支持情况。

### 模型在移动端进行推理优化的框架有了解过吗？移动端和在服务器的推理优化思路有何不同？移动端能用到的加速指令有了解过吗？

1. **TensorFlow Lite：** TensorFlow Lite是谷歌推出的移动端和嵌入式设备上的轻量级深度学习推理框架。它针对移动设备的硬件和资源特点进行了优化，支持量化、模型剪枝等技术。
2. **PyTorch Mobile：** PyTorch Mobile是PyTorch的移动端扩展，可以在移动设备上进行深度学习模型的推理。它提供了模型导出、量化和部署等功能。
3. **Core ML：** Core ML是苹果推出的移动端机器学习框架，支持将训练好的模型转换为移动端可用的格式。它可以在iOS和macOS设备上进行模型推理。
4. **NCNN：** NCNN是一个轻量级的高性能神经网络计算框架，专为移动端设备和嵌入式系统进行优化。它支持多种硬件平台，并且提供了模型转换工具和量化方法。
5. **Arm NN：** Arm NN是Arm推出的神经网络推理引擎，针对移动设备和嵌入式平台进行优化。它支持多种硬件平台和深度学习框架。
6. **TFLite Micro：** TensorFlow Lite Micro是针对嵌入式系统和微控制器的轻量级版本，用于在资源受限的设备上进行深度学习模型的推理。
7. **MNN（Mobile Neural Network）：** MNN是阿里巴巴推出的移动端深度学习推理框架，支持多种硬件平台和模型格式，提供了量化和自动融合等功能。

---

1. **计算资源限制：** 移动设备通常具有较小的计算资源，包括CPU和GPU的性能相对较低。因此，在移动端进行推理优化时需要考虑模型大小、计算复杂度和计算速度，以保证在有限资源下获得较好的推理性能。
2. **内存限制：** 移动设备的内存容量有限，因此需要优化模型的内存占用。这可以通过模型量化、模型剪枝、分片推理等技术来实现。
3. **功耗和热效应：** 移动设备的功耗和热效应是重要的考虑因素，高功耗会降低电池寿命，过高的温度会影响设备性能。因此，在优化时需要考虑如何降低功耗和热效应，例如选择较低功耗的算法、合理利用硬件加速等。
4. **用户体验：** 移动设备的用户体验至关重要，推理优化不应对应用性能和响应时间产生显著的负面影响。因此，推理优化需要平衡性能和响应时间，以提供良好的用户体验。
5. **网络连接：** 移动设备通常会在不稳定的网络环境中工作，因此在推理优化时需要考虑断开连接、重新连接等情况的处理，以保证推理过程的稳定性。

---

1. **ARM NEON：** ARM NEON是一种SIMD（单指令多数据）指令集，用于在ARM架构的处理器上进行并行计算。它可以加速向量运算、卷积等操作。
2. **Apple Core ML：** Apple的Core ML框架为iOS设备提供了专门的神经网络加速器，可以加速深度学习模型的推理。
3. **Android NNAPI：** Android Neural Networks API（NNAPI）是用于在Android设备上进行神经网络推理的API，可以利用硬件加速器执行模型。

### 移动端有哪些加速方法？

1. **模型量化（Quantization）：** 将模型参数和激活值从浮点数转换为整数，以减少计算和存储需求。量化可以显著减少模型的内存占用和计算开销。
2. **模型剪枝（Model Pruning）：** 移除模型中冗余的参数和连接，减少模型的大小和计算复杂度。剪枝后的模型可以在移动设备上更快地进行推理。
3. **深度可分离卷积（Depthwise Separable Convolution）：** 这种卷积操作将标准卷积分解为深度卷积和逐点卷积，减少了计算量，适用于轻量级模型。
4. **硬件加速器：** 移动设备上的硬件加速器，如GPU、NPU、DSP等，可以用于执行神经网络计算。将计算任务分配给适当的硬件加速器可以提高推理速度。
5. **异构计算（Heterogeneous Computing）：** 利用多种硬件加速器进行并行计算，根据任务需求分配计算资源，以提高整体性能。
6. **快速库和框架：** 使用针对移动设备优化的推理库和框架，如TensorFlow Lite、PyTorch Mobile等，以实现高效的推理。
7. **缓存优化：** 优化数据布局和内存访问模式，以利用移动设备上的缓存层次结构，减少内存访问延迟。
8. **动态图量化：** 在推理过程中，根据模型输入的分布来动态地量化模型，以适应不同的输入情况。
9. **流水线操作：** 将计算任务划分为多个阶段，在一个阶段进行计算的同时，准备下一个阶段的输入，以提高计算的吞吐量。
10. **并行推理：** 利用多个线程或进程在移动设备上并行地进行推理，以充分利用多核处理器。

### 为什么要将模型一部分推理优化放在移动端，全部放在服务器上不可以吗？

1. **延迟和实时性：** 在某些应用中，对于用户体验来说，实时性是至关重要的。将部分推理任务放在移动端可以减少网络传输延迟，从而更快地响应用户操作。
2. **带宽和网络：** 客户端到服务器的网络传输可能会面临带宽限制，特别是在移动网络环境中。将一部分推理任务在移动端完成可以减少数据传输量，降低网络开销。
3. **隐私和数据安全：** 涉及敏感数据的应用（如语音识别、人脸识别等）可能需要在设备上进行本地推理，以避免将数据传输到服务器，从而提高隐私和数据安全。
4. **离线支持：** 移动设备可能会在离线情况下运行，而服务器可能无法始终提供服务。在本地进行推理优化可以确保即使在没有网络连接的情况下也能进行模型推理。
5. **服务器负载分担：** 对于大规模部署的应用，将一部分推理任务放在移动端可以减轻服务器的负载，提高整体系统的可扩展性和性能。

### 自动驾驶上的推理框架有了解过吗？我们重点需要关注的指标有哪些？

即边缘侧推理引擎

1. **推理速度（Inference Speed）：** 自动驾驶系统需要实时感知和决策，因此推理速度是关键指标之一。推理速度要求在毫秒级别内完成，以保证车辆在实时环境中做出快速响应。
2. **延迟（Latency）：** 推理框架的延迟是指从输入数据传入到输出结果生成的时间间隔。低延迟是自动驾驶系统的关键要求，以确保车辆对环境变化做出及时反应。
3. **吞吐量（Throughput）：** 吞吐量指的是单位时间内可以处理的推理请求数量。高吞吐量可以支持处理多个感知任务，如物体检测、语义分割等。
4. **模型准确度（Model Accuracy）：** 自动驾驶的安全性和可靠性要求模型在不同情况下具有较高的准确度。推理框架需要确保在高速、复杂交通环境下的准确性。
5. **模型大小（Model Size）：** 模型大小直接影响内存占用和加载速度。较小的模型可以减少内存需求，并更快地加载到设备上。
6. **功耗（Power Consumption）：** 推理框架的功耗影响车辆的电池寿命和热效应。低功耗设计有助于延长自动驾驶车辆的使用时间。
7. **资源利用率（Resource Utilization）：** 推理框架需要合理利用硬件资源，如CPU、GPU、NPU等，以充分发挥硬件性能。
8. **实时性要求（Real-time Requirements）：** 自动驾驶系统对推理的实时性有严格要求，需要确保推理框架可以在预定的时间内完成任务。
9. **多任务支持（Multi-Task Support）：** 推理框架需要能够同时处理多个任务，如物体检测、行人识别、车道保持等。
10. **安全性和可靠性（Safety and Reliability）：** 自动驾驶系统必须保证在各种情况下的安全性和可靠性，包括异常情况下的模型行为和输出。

### 反向传播的原理，具体实现的源码有了解过吗？

数值微分 : 截断误差 - 随步长减小减小。 舍入误差 - 随步长减小增大

符号微分：科学计算和theano。表达式膨胀

基本表达式法：封装大多数的基本表达式及对应的微分表达式，通过库函数的方式提供给用户，用户在写代码时，需要手工分解程序为一系列的基本表达式，然后使用这些库函数去替换这些基本表达式。

自动微分

代码变换法（Source Transformation，ST）：提供对编程语言的扩展，分析程序的源码或抽象语法树（AST），将程序自动地分解为一系列可微分的基本操作，而这些基本操作的微分规则已预定义好，最后使用链式法则对基本操作的微分表达式进行组合生成新的程序表达来完成微分。TensorFlow，MindSpore等机器学习框架都采用了该方式。

（1）基于Tape的方式。该方式使用一个全局的”tape”去确保中间变量可以被获取到。原始函数被扩展为在前向部分执行时把中间变量写入到tape中的函数，在程序执行反向部分时会从tape中读取这些中间变量。除了存储中间变量外，OO中的tape还会存储执行的操作类型。然而因为tape是一个在运行时构造的数据结构，所以需要添加一些定制化的编译器优化方法。且为了支持高阶微分，对于tape的读写都需要是可微分的。而大多数基于tape的工具都没有实现对tape的读写操作的微分，因此它们都不支持多次嵌套执行反向模式的自动微分（reverse-over-reverse）。机器学习框架Tangent采用了该方式。

（2）基于闭包（closure）的方式。基于闭包的方式可以解决基于tape方式的缺陷。在函数式编程里，闭包可以捕获到语句的执行环境并识别到中间变量的非局部使用。因为这些它们是闭包里的自由变量，所以不需要再去定制化编译器优化方法。

直接反向传播

### 你了解哪些推理模型的调度方法？

### 推荐模型的结构有了解过吗？要部署一个大的推荐模型，应该如何将各个部分放在哪种硬件上部署？

### 计算图切分有了解过吗？如何应用于大模型推理？

计算图切分（Graph Partitioning）是一种将大型神经网络模型分割成多个子图，然后将这些子图分配给不同的计算设备或服务器进行推理的技术。这种方法有助于提高大型模型的推理性能和资源利用率，特别是在分布式环境中。以下是计算图切分应用于大模型推理的一般步骤和优势：

**应用步骤：**

1. **图划分：** 将整个大型计算图划分为多个子图。划分可以根据不同的准则，如节点数、边数、计算量等进行，以便均匀划分和平衡负载。
2. **设备分配：** 将划分得到的子图分配给不同的计算设备、GPU、CPU等。分配的目标是将计算量均匀分布，并充分利用各个设备的性能。
3. **通信优化：** 在分布式环境中，不同设备之间可能需要通信来共享数据。优化通信策略可以减少通信开销，例如通过本地通信或异步通信。
4. **子图推理：** 在每个设备上独立进行子图的推理。每个设备只需要处理分配给它的部分计算图，从而减小了单个设备上的计算负担。
5. **结果合并：** 如果需要，将不同设备上推理得到的结果合并为最终的推理结果。

**优势和适用场景：**

1. **性能提升：** 计算图切分可以充分利用多个设备的计算能力，加速大型模型的推理过程。
2. **资源利用：** 利用了多个设备的资源，避免了单一设备的瓶颈。
3. **分布式环境：** 特别适用于分布式环境，如数据中心、云平台等，可以通过多台服务器上的并行计算进行推理。
4. **大模型支持：** 对于参数量庞大的深度学习模型，通过切分可以降低单个设备的内存需求。
5. **实时性：** 在一些实时应用中，通过切分可以减小单次推理的计算量，从而提高实时性能。
6. **模型部署：** 对于移动端或边缘设备，可以将部分模型划分到设备上进行本地推理，减少对网络的依赖。

需要注意的是，计算图切分涉及到多个技术领域，如图划分算法、通信优化、并行计算等。实际应用中需要根据模型结构、硬件配置和推理需求进行综合考虑和调整。一些深度学习框架和工具提供了计算图切分的支持，可以简化实现过程。

### TensorFlow和Pytorch都用过吗？它们设计思路有何不同？有何优劣？如何添加自定义算子？

**TensorFlow和PyTorch的设计思路差异：**

1. **静态图 vs. 动态图：** TensorFlow采用静态计算图，首先定义计算图，然后执行计算。PyTorch采用动态计算图，允许在定义过程中直接执行操作，更容易进行调试和可视化。
2. **易用性 vs. 灵活性：** PyTorch通常被认为对于初学者更友好，因为它的动态图和直观的调试能力。TensorFlow的设计更加灵活，适合处理复杂的模型和优化。
3. **API一致性：** TensorFlow 2.x引入了Eager Execution，使其API更加与PyTorch类似，提高了易用性和可读性。

**TensorFlow和PyTorch的优劣：**

TensorFlow的优点：

* 强大的部署生态系统，如TensorFlow Serving和TensorRT，适用于移动设备和嵌入式系统。
* 静态图可以优化计算图，有利于在生产环境中获得更好的性能。
* 丰富的社区支持，拥有广泛的文档、教程和模型资源。

TensorFlow的缺点：

* 初始学习曲线较陡峭，可能需要时间来适应其复杂的API和概念。
* 部分API不够直观，可能会导致初学者难以理解。

PyTorch的优点：

* 动态图设计使其易于调试和可视化，适合于实验和研究。
* 直观的API设计，易于上手和使用。
* 深度学习研究者和实验室广泛采用，具有较活跃的社区。

PyTorch的缺点：

1. 部署时相对不如TensorFlow灵活，尤其在移动设备和嵌入式系统上。
2. 某些情况下，动态图可能在生产环境中导致一些性能问题。

**自定义算子的添加：**

* 在TensorFlow中，您可以通过定义操作的计算方式并将其注册到TensorFlow框架中，从而添加自定义算子。
* 在PyTorch中，您可以编写自定义函数或操作，然后将其封装为PyTorch的Autograd Function，以便支持自动求导。

## 算法题

手写CUDA kernel几乎每场面试都会考，面试官会以写出来的第一个版本为准，一步步问继续优化的方法，在这个期间会结合高性能计算的基础知识来考察，从这个过程中能了解到对体系结构以及优化方法的了解程度。leetcode不一定有，但是遇上了基本上都是hard。两类算法题都要准备。

下面是常见的一些问题：

### gemm

https://zhuanlan.zhihu.com/p/410278370

Strided Batched GEMM

https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/

### self-attention

https://www.bilibili.com/video/BV1dt4y1J7ov

### BN ，layernorm 层有什么用，具体怎么算的？

    随着训练，激活结果的分布可能会从 0附近偏移，导致梯度消失，梯度爆炸，训练结果不好
    https://www.bilibili.com/video/BV1DM4y1w7J4

BN对不同batch的chanel做，layernorm对同一batch的channel之间逐像素做

### softmax有什么用，怎么做的

   https://www.bilibili.com/video/BV1cP4y1t7cP 最大熵原理

1. 矩阵乘:
2. 矩阵转置: 访存密集型算子的处理
3. 一维reduce-sum：重点是如何处理bank confict
4. 二维reduce-sum
5. 卷积
6. 将单stream改成多stream

以矩阵乘法为例说明一下一个典型的面试流程，下面以A表示面试官，B表示面试者。

A：写一个矩阵乘法吧，并将main函数中具体调用给写清

B: （写了一个最naive版本的矩阵乘）

A: 目前这个程序有什么问题，能进一步优化吗？

B : 目前访存性能比较低，可以采用矩阵分块并且使用上shared memory优化，并解释一下这样做的原理。

A：可以具体计算一下优化前后的计算访存比，来具体说明这一部分提升了多少。并写一下优化后的程序。

B: 通过计算优化了.....

上述对话会重复几轮，在后面几轮可能面试官不会再要求将每一版程序都写出来了，重点在于讨论优化思路，并且在讨论的过程中发散地问一点CUDA的知识考察理解的深度。

### reduce 优化：

分支发散：tread 集中

1. banck 冲突： thread 连续访问 bank
2. 初始化闲置：load and add

最后一次循环不需要同步和条件判断：展开

完全展开

一个线程计算更多，lunch开销，

https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

### transpose

https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

https://segmentfault.com/a/1190000043451674

## 一些比较零碎的问题

### 1. 卷积的三种加速计算方式，im2col+GEMM & Winograd & FFT，各自有何优缺点，cuDNN中有哪些实现？

1. **im2col + GEMM (General Matrix Multiply)：**
   这种方法将卷积操作转化为矩阵乘法（GEMM）的操作。首先，通过将卷积核展开为列向量，将输入数据转换为一个大的矩阵，然后使用矩阵乘法来计算输出特征图。
   **优点：**

   * 可以充分利用现代CPU和GPU的高效矩阵乘法实现。
   * 简化了卷积操作的实现，容易并行化。

   **缺点：**

   * 转换操作（im2col）可能会增加内存消耗。
   * 对于小的卷积核和输入，转换操作可能会带来较大的计算开销。
2. **Winograd卷积：**
   Winograd卷积是一种通过变换输入数据和卷积核，将卷积操作转化为小规模矩阵乘法的方法。这种方法通过减小矩阵的规模来提高计算效率。
   **优点：**

   * 对于较小的卷积核，可以在一定程度上减少计算量。
   * 可以利用矩阵乘法的高效实现。

   **缺点：**

   * 实现较复杂，需要进行Winograd域变换和逆变换。
   * 对于较大的卷积核和输入，可能无法获得显著的性能提升。
3. **FFT卷积：**
   快速傅里叶变换（FFT）可以将卷积操作转化为频域的乘法操作。将输入和卷积核进行FFT变换，然后执行频域的元素乘法，最后进行逆FFT变换得到卷积结果。我测过的数据是当卷积核的边长大约等于图像边长的1/3以上，傅里叶变换才会有时间优势。
   **优点：**

   * 适用于大卷积核，因为FFT计算复杂度相对较低。
   * 可以通过FFT库实现高效的频域操作。

   **缺点：**

   * 对于小卷积核，可能存在额外的计算开销，因为FFT变换本身的计算也需要时间。
   * 实现相对复杂，需要对FFT库的调用和频域操作进行处理。
   * 内存带宽需求增大：变换产生的复数、变换后变大的kernel都会增大内存带宽需求。
   * 计算更复杂：如果没有专门的计算单元或者指令的话，一次[复数乘法](https://www.zhihu.com/search?q=%E5%A4%8D%E6%95%B0%E4%B9%98%E6%B3%95&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A280445640%7D)实际上需要四次实数乘加。

### 2. 数字信号的采样定理、熵 & 交叉熵 的含义 & 计算公式

   x个球里面摸球 的 信息量 ： logx 熵：事件的总信息量，交叉熵： 预期概率 P 实际概率 Q QlogP

### 3. 还记得KKT条件吗？写程序求解一个非线性方程，并说明具体用到的优化方法。

### 4. 脑洞问题：如何从编码的角度进行模型压缩？

### 5. 如何将你研究生阶段的成果应用到我们的产品中？

### 6. 给了一个TF 模型的profile，找出里面的bottle neck，提出如何改进这个模型的性能的方法。

   nsight sys，查看kernel空闲处的原因，dataload？延迟掩藏？带宽上限？算力上限？

   nsight compute

### 7. MIPS流水线有几级？分别是哪些组成部分？

1. **取指令（Instruction Fetch，IF）：** 从指令内存中获取下一条指令，并将其送入流水线。
2. **指令译码（Instruction Decode，ID）：** 对取出的指令进行解码，确定操作类型、寄存器和立即数等。
3. **执行（Execution，EX）：** 根据解码后的指令，执行计算操作，如算术逻辑运算、内存地址计算等。
4. **访存（Memory Access，MEM）：** 如果指令需要访问内存（如加载/存储指令），在此阶段进行内存操作。
5. **写回（Write Back，WB）：** 将执行结果写回寄存器文件。

### 8. 说一下transformer的具体结构，如何加速transformer进行推理？

1. https://www.bilibili.com/video/BV1dt4y1J7ov
2. https://www.bilibili.com/video/BV1ih4y1J7rx

### 9. attention的计算公式，写一下tf里面对应的代码

### 10. 马尔科夫链简单知识

### 11. 一道较难的概率题

## 推荐参考资料

1. 《通用图形处理器设计:GPGPU编程模型与架构原理》：CUDA、GPU体系结构、PTX、TensorCore等GPU知识大杂烩，CUDA相关面试问题标答。对于GPU的硬件体系结构有较深入的介绍，虽然比较难懂，但是这一部读完后会对CUDA编程模型以及为什么要采用一些特定的优化方法有更深入的理解。
2. 官方文档《CUDA Programming Guide》 & 《CUDA Best Practice Guide》: 不解释，必读。
3. 《大规模并行处理器程序设计》：入门最佳，没有之一。其中第二部分对于CUDA中常见的计算Pattern做了分析，几乎可以应付所有的面试中的kernel编程，至少能答出80%，至于更深入地优化方法需要再花时间去研究。
4. 《[机器学习系统：设计和实现](https://openmlsys.github.io/)》：介绍了ML Sys这一领域的所有方面的基础知识，可以从一个整体的层面对机器学习系统的组成部分、每个部分的重点技术有较好的把握。这本书的框架主要以MindSpore为例，所以在整体读完后，需要结合自己比较熟悉的框架进一步仔细理解。
5. 《深度学习进阶：自然语言处理》：只用numpy实现NLP模型，可以作为阅读深度学习框架源码的first course，会对AI模型中的底层实现细节有很好的理解。
6. 《分布式机器学习：理论、算法与实践》：可以对分布式训练有大致的了解
7. 《AI编译器开发指南》：深度学习编译器相关的介绍，重点在TVM。

建议： 1 ~ 4必读，这是所有领域的基础知识，5 ~ 7需要根据个人的研究兴趣和方向有选择性地深入阅读。

# 帅哥

## 卷积的流程？

nhwc 的 layout 易于对特征图分块尺寸和指令对齐，不需要memcpy
供调用的conv汇编级别指令是对所有in_channel做一个filter的卷积
需要对out channel循环。
？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？完善

conv算子：

1. 硬件有conv操作，channel 需要64 对齐，第一个conv只有3channel，只发挥了3/64的性能通过trans将channel做到48(3x4x4)，可以发挥48/64的性能。

如何将3->27的，额外的开销？dma, 延迟掩藏

conv是一个汇编指令

2. 可变shape，需要在算子前面加标量计算，因为要实时演算形状，拆分方式，循环次数等等的计算策略参数，串行执行导致硬件空转性能大幅度下降。通过一点一点注释代码+计时定位。然后通过 调整指令顺序，将没有依赖的指令提前，用数据加载掩藏标量计算。硬件用verilog验证波形符合猜想。

## Nsight compute

https://www.bilibili.com/video/BV13w411o7cu

## 通用优化



合理使用原子操作 https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

避免warp分支

减少数据传输

合并访存

双缓存技术

sharedmem 避免 banck conflect （跨步访问，空一错开）

面向warp编程

gridsize 参考 SM 数量

### 优化内存与显存传输效率


* 避免 global mem 的重复访问
* 使用Pinned(page-locked) Memory提高传输速度
* 通过在不同的Stream里同时分别执行kernel调用及数据传输，使数据传输与运算并行。（注意default stream的坑 ^[[1]](https://zhuanlan.zhihu.com/p/570795544#ref_1)^ ）
* 尽量将小的数据在GPU端合成大块数据后传输
* 有些情况下，即使数据不太适合使用kernel处理，但如果为了较低的算法latency，也可权衡传输代价后使用kernel处理数据
* 注意PCI-e插口的通道个数

### 优化Kernel访存效率

* **提高Global Memory访存效率**

1. 对Global Memory的访存需要注意合并访存(coalesced )。^[[2]](https://zhuanlan.zhihu.com/p/570795544#ref_2)^
2. warp的访存合并后，起始地址及访存大小对齐到32字节
3. 尽量避免跨步访存
4. 8.0及以上的设备可以通过编程控制L2的访存策略提高L2命中率。

* **提高Shared Memory的访存效率**

1. shared memory由32个bank组成
2. 每个bank每时钟周期的带宽为4字节
3. 连续的4字节单元映射到连续的bank。如0-3字节在bank0，4-7字节在bank1……字节128-131字节在bank0
4. 若warp中不同的线程访问相同的bank，则会发生bank冲突(bank conflict)，bank冲突时，warp的一条访存指令会被拆分为n条不冲突的访存请求，降低shared memory的有效带宽。所以需要尽量避免bank冲突。
5. CUDA 11.0以上可以使用*async-copy* feature^[[3]](https://zhuanlan.zhihu.com/p/570795544#ref_3)^

### 优化线程级并行

访存瓶颈时可以单线程处理多数据，提高实际带宽。
occupancy calculator
### 指令级优化

* **提高计算访存比**

GPU执行计算时，需要LDS、LDG等指令先将数据读入寄存器，再进行计算，最后通过STS、STG等指令将数据保存下来。

以矩阵乘法为例，先进行矩阵分块，最终拆解为每个线程计算MxK,KxN的两个小矩阵的乘法：

若两小矩阵为M=2,N=2,K=1，即2x1;1x2,最后得到2x2的矩阵作为结果。则读入4个float需4条指令，计算指令也是4条，计算访存比4/4=1；

若两小矩阵为M=8,N=8,K=1，即8x1;1x8,最后得到8x8的矩阵作为结果。则读入16个float，需读取指令16条，计算指令8x8=64条，计算访存比64/16=4；若使用向量读(float4)每条指令读入4个float，则读取指令仅4条，计算访存比64/4=16

提高计算访存比，可以让GPU的更多时钟周期用于进行计算，相对的进行数据IO占用的时钟周期更少。

* **提高指令级并行**

指令级并行基本原理：

* 现代不论是CPU还是GPU，指令的执行都是通过流水线进行的，流水线分为多个stage，即一条指令执行完成需要每个stage的工作都执行完成。而一个时钟周期并不是完成一条指令执行的所有时间，而是每一个stage完成当前工作的时间。流水线可以同时执行多条指令的不同阶段。
* 当后续指令的执行需要依赖前面指令的结果写回寄存器，我们说出现了寄存器依赖。此时后续指令需要等待第前面指令结果写回寄存器才能执行，若后续指令执行时前面指令结果尚未写回寄存器，流水线会失速（stall），此时warp scheduler开始切换到其它eligible warp，若无eligible warp，则SMSP将会空转。
* 若后续指令不依赖前面指令的结果，则即使前面指令未执行完毕，后续指令也可以开始执行。特别的，即使前序指令是一条耗时几百周期的LDG(全局内存读取)指令或耗时几十周期的LDS（共享内存读取）指令，只要后续一系列指令不依赖读取回来的数据，后续一系列指令可以正常执行而不必等待该LDG/LDS指令执写回寄存器。

通过以下方式，可以提高指令级并行，在线程级并行达不到较好效果的情况下，进一步提高程序性能：

* 数据预取（Prefetch）：数据1已读取到寄存器，使用该数据1计算前，先将后续数据2的读取指令发射，再执行一系列数据1的处理指令；这样数据1的处理和数据2的读取在流水线上同时执行着。当数据1处理完成，需要处理数据2时，可以确保数据2已经存在于寄存器中，此时类似的将数据3的读取和数据2的处理同步执行起来。
* 指令重排：在存在寄存器依赖的指令间插入足够的其它指令，使得后续指令执行时，前面计算指令的结果已写回到寄存器。从CUDA C层面有意识地提供一些语句间的并行性，nvcc编译器可以一定程度上自动进行指令重排。若对nvcc重排结果不满意需要自己重排时，官方尚未开放SASS汇编器，目前只存在一些第三方SASS汇编器工具 ^[[5]](https://zhuanlan.zhihu.com/p/570795544#ref_5)^ 。
* 
* **提高Register的效率**

1. Register File也存在bank冲突，但在CUDA C层面上没有直接办法进行物理寄存器控制。
2. 可以通过SASS汇编器，人工进行指令寄存器分配，以尽量消除register bank conflict。
3. 可以通过SASS汇编器，为寄存器访问添加reuse标记，以尽量消除register bank conflict。

### 使用TensorCore进一步加速矩阵运算^[[6]](https://zhuanlan.zhihu.com/p/570795544#ref_6)^

TensorCore可以用来快速进行D=A*B+C矩阵运算，提供 `load_matrix_sync`， `store_matrix_sync`， `mma_sync` 等API。


# 太有意思了

结论： 浮点数加上超过当前exp情况下能表示的最小值，会舍入到表达能力的下一个数  
结论： template DATA_TYPE 的 kernel 没有意义，因为 shared mem 必须为 extern 会导致同名但是类型不同的变量  
结论： Nsight compute bug，bank conflict 还记录了其他 stall  
https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric