# 你的缺点？（怎么改进和克服）
## 原则
1.选择一个不会妨碍你成功的缺点。  
2.要诚实，选择一个真正的弱点。  
3.提供一个例子，说明你是如何努力改善自己的缺点，或者学习一项新技能来解决这个问题的。  
4.表现出自我意识和向他人寻求成长所需资源的能力。  
5.不要自大，也不要低估自己。
## 例子
### 懒惰和拖延
不考普遍的情况，仅对于我自己来说，我认为懒惰和拖延的唯一原因就是动机不足，而动机不足是因为对待办的事思考的不够透彻，处于一个比较迷糊的状态。  
对事情进行比较深入的思考，之后一般有几种结果：  
1.认识到待办事项的非常必要，紧急，划算，有趣，找到足够的动机和动力去做了。就不存在懒惰和拖延的问题了。  
2.经过思考认为确实是不必要的，低优先级的，那就降低先级，有空了，或者有兴致了再处理  
3.确认是错误的事情，别瞎搞  

最开始对什么事情都需要经过一番思考，很慢但是很理智。思考的多了就会形成一种类似“自动驾驶”的机制也就是“惯性思维”，“惯性思维”很有用，可以帮我们快速的作出不差的决定。但也要时刻警惕是不是掉入自己惯性思维的陷阱了  

工作中，对于待办事项，搞清楚来龙去脉是最重要的，理解的深入才能有开展的动力，才能开展的正确，高效。如果有说不清的情况，就是不能深入理解（别人处于某些原因，不想让我理解）。又必须做那就只能是，尽力

### 不太喜欢聊八卦
虽然工作上的沟通表达很清晰，但是很少和聊八卦，其实八卦是信息交换很的活动，可能是性格原因，或者工作太忙，就没时间没兴趣聊天了，话都和同事说干净了

# 你工作中遇到最大的的挑战是？你是如何克服的？
## 原则
1.搜索你以前面临的挑战  
2.根据职位描述定制你的答案  
3.具体说明为什么它们是挑战  
4.诚实回答问题  
5.以积极的眼光来呈现你的挑战  
6.必要时使用非专业的例子。  
## 例子
最大的可能谈不上，就是最近解决的一个技术上的问题。  

### 多进程通信处理 SIGINT
现象：概率性出现的随机失败问题。  
事后定位问题根因：python multiprocessing 多进程通信。SIGINT 会导致阻塞式调用的 manager.queue.get 行为异常。具体是 queue.get 阻塞式调用时如果收到 SIGINT 不会马上打断，而是拿到对象再打断，而且异常不会被 catch 到，因为进程已经结束了。导致本应该常在的对象丢失。
回溯：最开始出现是因为负载的上升（由于一个待测产品的新增已知问题）而触发到了之前一直没有被发现的这个测试代码bug，也就是几乎没修改代码，仅修改了任务配置的情况下，原来正常运行的测试就出现奇怪的随机失败。  
稳定复现和问题定位：加打印，怀疑到这个方向之后构造必现情况（仅用一个常在对象）。查找对象消失具体位置和原因，google，python 文档，发现这样一条
Warning: If a process is killed using Process.terminate() or os.kill() while it is trying to use a Queue, then the data in the queue is likely to become corrupted. This may cause any other process to get an exception when it tries to use the queue later on.
实际也确认到确实是这种情况。  
解决：  
1.使用其他信号应该也是不行的，因为子进程阻塞中。  
2.重构 timeout 机制，改为进程内部timeout，引起一些其他问题：何处 timeout多久？等等  
考虑率工作量和稳定性，workaround，改为非阻塞式调用，但是场景需要阻塞，外层套while  
添加此场景的precheckin test





# 领导 朋友 同事 如何评价你？