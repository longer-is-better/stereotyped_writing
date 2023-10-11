# 部门情况？
软件部，cnml算子组，20人，按照算子分任务，开发维护和优化，之前算子组有人员裁减，长打交道的是推理引擎和runtime组，runtime组长比较强硬，和这组组员交涉的时候要提前充分准备，语气要谦和坚定，就事论事




# 软件栈？

算子组 cnml cnnl cncv
CNML主要是针对MLU220 MLU270系列硬件的算子库，而CNNL是针对MLU290 MLU370系列的。主要区别如下：
1. CNML算子库是不支持shape可变的，CNNL是支持shape可变的。  
2. MLU220/270用CNML主要是面向模型融合推理，而MLU370用CNNL是支持训练的。
cncv 视觉，图像算子库。


框架组，TF pytorch... 框架对接  

推理引擎 magicmind：  
图 解析，优化，编译， runtime  

cncl 通信
runtime: cuda runtime
编译器 bangC




# 如果你是领导，你如何管理团队
https://www.zhihu.com/question/282778286  


# 规划？
继续走技术路线，没事的时候会看一些管理相关的内容，作为消遣了