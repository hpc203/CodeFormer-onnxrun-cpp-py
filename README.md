# CodeFormer-onnxrun-cpp-py
使用ONNXRuntime部署CodeFormer图像清晰修复，包含C++和Python两个版本的程序。
CodeFormer是一个很有名的图像超分辨率模型，它是NeurIPS 2022会议论文。

训练源码在https://github.com/sczhou/CodeFormer

onnx文件在百度云盘，下载链接：https://pan.baidu.com/s/18g918VdcyODdVcSOCLutOA?pwd=ixre 
提取码：ixre

在编写完程序后，测试了多张图片，发现如果输入图片本身是清晰的图片，那么经过CodeFormer的推理之后，
得到的图片反而变形走样了，使得图片变得不清晰了。因此，一套合理的程序是，在执行CodeFormer之前，
对输入图片做一个模糊检测，如果判定图片是模糊的，才能输入到CodeFormer。试验分析，可以看我的
csdn博客文章 https://blog.csdn.net/nihate/article/details/112731327
