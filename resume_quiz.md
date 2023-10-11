# 被问到入职意向
我最想来贵司了

# 推理引擎的的输入输出？
build_time：
输入是：开源框架模型ptyorch tensorflow onnx...
输出是：序列化的 MM model
runtime：
网络输入输出

# python用过哪些库？
os system re
threading multiprocessing  
numpy pandas  
paramiko  
pytorch  
huggingface  

# 如何测试算子
magicmind 单算子测试框架即CS框架，C/S指client/server，单算子测试框架目前包括两部分，分别是真值生成以及测试。生成真值的过程是采用web 轻量级框架 Flask 生成服务：输入描述模型以及输入数据的基础json，返回CPU推理结果及对应的框架模型到指定路径，供测试程序在MLU上进行推理，最后将MLU的推理结果与CPU的推理结果进行对比，采用寒武纪算法组的精度标准。按大类分为两种测试，type = op是使用builder_api构建的测试， type = network是使用parser构建的测试。  

Pytorch 单算子测试框架方案整体设计:  
Pytorch 单算子测试框架复用magicmind C/S框架结构  
Server复用magicmind框架Server端的设计：  
通过Flask相应client的request，来调用算子模型脚本在GPU上生成Golden,并采集GPU性能信息  
将模型文件、输入数据、以及真值打包供client下载  
Client重新设计，使用python编写替代现有magicmind client端的实现单一、批量算子测试  
发送包含有测列相关信息request到server，来下载相应的server端生成的包  
解压并将输入设置到模型脚本，在MLU上运行，获得结果，并采集性能数据  
对比GPU与MLU的结果并检验误差(<0.005)  
生成精度误差报告  
## 确认json测例场景的完备性
json测例主要从以下几个方面覆盖测试场景（前提是算子自身特性支持），有些场景算子server端真值不支持时，建议使用其他测试方法。
(注意，云测如mlu370_s4、mlu370_x4、mlu370_x8、mlu365_d2等不需要添加qint16_mixed_float16、qint16_mixed_float32精度模式;端侧如ce3226、ce3225、sd5223不支持force_float32、forece_float16精度模式)
* 输入的type
* 输入的维度
* 参数的组合变化
* 输入为0元素
* 输入为标量
* evaluation精度的设置是否符合[MM算子精度问题及限制(包含tfu)](http://wiki.cambricon.com/pages/viewpage.action?pageId=59746786)规范
* 边界值
* 算子的多次推理(可变形状,固定形状)
* kernel capture的支持
* 算子裁剪
* 支持自动融合