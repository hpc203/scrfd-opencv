# scrfd-opencv
使用OpenCV部署SCRFD人脸检测，包含C++和Python两种版本的程序实现，本套程序只依赖opencv库就可以运行， 从而彻底摆脱对任何深度学习框架的依赖。

SCRFD是一个FCOS式的人脸检测器，2021年5月发出来的，SCRFD 是高效率高精度人脸检测算法，速度和精度相比其他算法都有提升。
你的机器里只要安装里OpenCV库，就能运行本套程序。C++版本的主程序是main.cpp，Python版本的主程序是main.py。
程序输出检测到的人脸矩形框和5个关键点
