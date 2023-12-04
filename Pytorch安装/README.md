# 常用命令

1. 更新pip

```shell
pip install --upgrade pip
```

2. 查看所有环境

```shell
conda env list
```

3. 创建新的环境

```shell
conda create -n 环境名称 python==版本

# 添加镜像加速
conda create -n 环境名称 python==版本 -c 镜像地址

# 持久添加通道
conda config --add channels 通道地址

# 删除通道
conda config --remove channels 通道地址

# 查看配置文件中有哪些通道
conda config --get
conda config --show


# 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# 北京外国语大学镜像：https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
# 阿里巴巴镜像：https://mirrors.aliyun.com/anaconda/pkgs/main
```

4. 删除虚拟环境

```shell
conda remove -n 环境名称 --all
```

5. 进入/退出环境

```shell
conda activate 环境名称
conda deactivate
```

6. 查看已安装的库(conda/pip)

```shell
conda list
pip list
```

7. PyCharm中Terminal（终端）的设置

- File -> Settings -> 搜索Terminal -> 更改Shell path

8. 给下载的项目进行环境配置

- 注意`requirements.txt`文件（用于存储运行项目所需要的包），打开后PyCharm可以自动检测哪些包没有安装
- 或者：

```shell
cd 项目所在位置
pip install -r requirements.txt
```

# Windows下安装PyTorch（CPU版本）

## Step1：安装Anaconda

- 官网：https://www.anaconda.com/
- 选择Products -> Anaconda Distribution
- 点击Download，可以下载最新版本
- 注意：安装路径最好全英文

## Step2：创建虚拟环境

```shell
conda create -n 环境名称 python==版本
```

- 推荐：python==3.8

## Step3：利用conda或pip安装PyTorch

### 方式一：conda✨

- 在上一步创建的虚拟环境中安装PyTorch

```shell
conda activate 环境名称
```

- 安装PyTorch，需要安装pytorch，torchvision，torchaudio三个包
- 从官网（[PyTorch](https://pytorch.org/)）采用命令行下载：

```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

- 添加镜像源：

```shell
conda install pytorch torchvision torchaudio cpuonly -c 镜像地址


# 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/
# 北京外国语大学镜像：https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/win-64/
# 阿里巴巴镜像：https://mirrors.aliyun.com/anaconda/cloud/pytorch/win-64/
# 南京大学镜像：https://mirrors.nju.edu.cn/pub/anaconda/cloud/pytorch/win-64/
```

### 方式二：pip

- 从官网（[PyTorch](https://pytorch.org/)）采用命令行下载：

```shell
pip3 install torch torchvision torchaudio
```

- 或者：下载PyTorch Package后，利用`pip install 路径地址`安装（https://download.pytorch.org/whl/torch_stable.html）

## Step4：验证是否安装成功

```python
import torch

torch.cuda.is_available()  # 返回False，则安装成功
```

> 安装旧版本的PyTorch：
>
> [Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)
>
> 注意：安装速度较慢时，换镜像通道！
>
> - 我们可以利用`conda search pytorch==版本 -c 镜像通道`来验证镜像通道中是否有这个版本的PyTorch！

# Windows下安装PyTorch（GPU版本）

## Step1：安装Anaconda

- 官网：https://www.anaconda.com/
- 选择Products -> Anaconda Distribution
- 点击Download，可以下载最新版本
- 注意：安装路径最好全英文

## Step2：创建虚拟环境

```shell
conda create -n 环境名称 python==版本
```

- 推荐：python==3.8

## Step3：GPU和CUDA准备工作

- 首先，确定自己显卡的算力
- 确定自己的可选择CUDA Runtime版本
- 确保自己的<b>CUDA Driver版本 >= CUDA Runtime版本</b>

> 具体步骤：
>
> 1. 确定显卡型号
> 2. 确定显卡算例：[CUDA - 维基百科，自由的百科全书 (wikipedia.org)](https://zh.wikipedia.org/wiki/CUDA)
> 3. 确定可以使用的CUDA Runtime版本
> 4. 查看自己的驱动CUDA Driver Version
>
> ```shell
> # 右上角：CUDA Version
> nvidia-smi
> ```
>
> 5. 最终根据：CUDA Driver Version >= CUDA Runtime Version，确定适用的CUDA Runtime版本

:star:上述过程过于繁琐，真实实践中，我们一般这样做：

- 先去装显卡驱动的最高版本（英伟达官网：nvidia.cn -> 驱动程序）
- 去PyTorch官网选择CUDA版本小于显卡驱动的CUDA版本即可

## Step4：利用conda或pip安装PyTorch

### 方式一：conda✨

- 在Step2创建的虚拟环境中安装PyTorch

```shell
conda activate 环境名称
```

- 安装PyTorch，需要安装pytorch，torchvision，torchaudio三个包
- 从官网（[PyTorch](https://pytorch.org/)）采用命令行下载：（以CUDA Runtime==11.3为例）

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

- 添加镜像源：

```shell
conda install pytorch torchvision torchaudio -c 镜像地址
conda install cudatoolkit=11.3 -c 镜像地址

# pytorch，torchvision，torchaudio镜像地址：
# 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/
# 北京外国语大学镜像：https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/win-64/
# 阿里巴巴镜像：https://mirrors.aliyun.com/anaconda/cloud/pytorch/win-64/
# 南京大学镜像：https://mirrors.nju.edu.cn/pub/anaconda/cloud/pytorch/win-64/

# cudatoolkit镜像地址：
# 清华镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# 北京外国语大学镜像：https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
# 阿里巴巴镜像：https://mirrors.aliyun.com/anaconda/pkgs/main
```

### 方式二：pip

- 使用`pip3 install torch torchvision torchaudio --extra-index-url https://downloads.pytorch.org/whl/cu113`
- 或者本地安装：
  - <b>用官网指令时会显示下载地址，复制下载地址到迅雷下载</b>
  - 如果本地安装完成，使用`pip install 把本地下载好的文件拖进来`
  - 注意：此时torch已经安装完成，但是torchvision和torchaudio还未安装，只需要再运行一次官网的指令`pip3 install torch torchvision torchaudio --extra-index-url https://downloads.pytorch.org/whl/cu113`即可，由于torch已经安装完成，不会进行重复安装，torchvision和torchaudio安装包较小，下载速度较快，使用官网安装方式即可！

## Step5：验证是否安装成功

```python
import torch

torch.cuda.is_available()  # 返回True，则安装成功
```

> 安装旧版本的PyTorch：
>
> [Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)
>
> 注意：安装速度较慢时，换镜像通道！
>
> - 我们可以利用`conda search pytorch==版本 -c 镜像通道`来验证镜像通道中是否有这个版本的PyTorch！
