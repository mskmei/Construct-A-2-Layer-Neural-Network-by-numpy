# construct a 2-layer neural network by numpy
本项目实现：除了加载数据与保存模型，仅仅使用numpy构建bp神经网络。一共有三个文件，分为神经网络类的构建，寻找最优参数及可视化与加载并验证模型

1、训练步骤：
    将文件三个文件“m_nn.py”、“find_para.ipynb”、“load_and_test.ipytnb”放入同一个文件夹下方，打开“find_para.ipynb”，并直接运行整个文件（本人是在jupyter notebook中运行的），程序会找出相对最合适的参数组合，并保存模型到当前文件夹，名字为“nn.pkl”。至于详细的关于参数的交叉比较与各个过程的可视化请详细见“find_para.ipynb”已经有的输出与批注。
    进而可以打开“load_and_test.ipytnb”，确保“nn.pkl”已经在同一个文件夹下，然后可以直接运行，可以求得测试过程的精确度。

2、直接使用现有模型进行测试：
    我也准备已经训练好的模型，可以直接下载使用。请点击链接 https://drive.google.com/file/d/17uzZ3Ol1OWh_3fgQH37WAYOutVTdMCRB/view?usp=sharing ，直接下载保存在网盘中的现有文件“nn.pkl”，并将该文件放入“load_and_test.ipynb”同一文件夹中，便可以直接运行“load_and_test.ipynb”，不需要下载其余的两个文件。
    
3、注意：
(1)加载数据集使用到了tensorflow.keras，请确保已经安装了相应环境与包；
(2)不能轻易改动文件名，包括保存的模型的名字；
(3)确保所有文件放在同一文件夹中。
