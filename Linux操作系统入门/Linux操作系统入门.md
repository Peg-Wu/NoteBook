## 1. 文件夹操作

cd：切换文件夹

mkdir：创建文件夹

cp：拷贝

mv：移动/重命名

rm：删除文件/夹

rmdir：删除**空文件夹**

history：保存历史命令

‘tab’自动补全路径，可提高输入效率

## 2. 查看文件内容

cat，more，less，head，tail都可用来查看文件内容

cat主要用于文件拼接

head和tail查看文件头部和尾部内容

more（只能向后查看）和less（向前向后查看均可）支持分屏查看，查看中间可按q键退出

## 3. 查找文件

locate和find均可用来查找文件

locate快，简单，适合初学者

find递归搜索，更强大，更灵活，也更复杂

## 4. 查看文件树

"Everything is a file." 整个Linux系统只有一个文件树

man和info可用于查看帮助文档

## 5. 输出（stdout和stderr）重定向

\>      会覆盖原文件内容

\>>   添加（append）到文件结尾

> ✨
>
> 1：stdout    标准输出
>
> 2：stderr     标准错误

想要将标准错误也定向到文件中，可执行以下命令：

```bash
ls xxx 1> temp.out 2> temp.err

# 不想要标准错误，/dev/null是一个特殊的设备/文件，相当于Linux系统中的黑洞，将所有输入给它的内容抛弃
ls xxx 1> temp.out 2> /dev/null

# 想要将标准输出和标准错误放在同一个文件下
ls xxx > temp.txt 2>&1
# 2>&1 将标准错误重定向到标准输出
```

## 6. 输入重定向

- 标准输入（stdin）通常是键盘

<     输入重定向

```bash
# bc可用于进行数学计算
bc < temp.txt

# temp.txt用于参数传递，temp.out用于保存输出结果
bc < temp.txt > temp.out
```

## 7. 查看磁盘和内存空间

df -h

- df （disk free）：查看磁盘空闲空间

- -h （human-readable）

du -h -d 1 .

- du（disk usage）
- -h （human-readable）：查看磁盘使用空间
- -d 1 （depth=1，只展示当前文件夹下的文件和子文件的占用情况，默认是recursive[递归]展示所有文件）

free -h：查看空闲内存

## 8. 查看进程

ps -ef

- ps （peocess）
- -l （long，长信息）
- -f（full，更多信息）
- -u （user，查看特定用户的进程信息）
- -e（everything，查看所有进程信息）

kill 进程号：结束进程

killall -u：结束指定用户的所有进程

top：实时查看系统的运行状态，按q退出

## 9. 环境变量配置

echo $PATH

- PATH环境变量：在这些目录下查找文件无需加路径
- 添加环境变量：PATH=$PATH:ADD_PATH
- 添加全局环境变量：export PATH=$PATH:ADD_PATH
- printenv：打印所有环境变量

unset：取消环境变量

## 10. 压缩和解压

gzip和gunzip

> *.gz通常是gzip算法的压缩文件

tar -zcvf 压缩之后的文件名称 待压缩的文件

tar -zxvf 待解压的文件 -C 解压到特定文件夹

> ✨
>
> -z表示使用gzip算法，压缩后的文件后缀为.tar.gz / .tgz
>
> -j表示使用bzip2算法，压缩后的文件后缀为.tar.bz2 / .tbz
>
> -c表示compression（压缩）
>
> -x表示解压

## 11. 管道操作pipe "|"

cmd1 | cmd2：**cmd1运行的输出作为cmd2的输入**

==通过管道，多个命令可以方便地组合使用（Do one thing at a time）==

## 12. 配置个人工作环境

export PATH=$PATH:ADD_PATH          该命令在每次退出系统后，环境变量设置就消失了，要直接保存环境变量应该怎么做？

> `~/.bashrc`和`~/.bash_profile`文件存储个人环境变量等配置信息，使用source命令（.）即时运行更新

- 在`~/.bashrc`文件最后加`PATH=$PATH:ADD_PATH`，之后运行`source ~/.bashrc`或`. ~/.bashrc`
- 添加`alias lr="ls -lrth"`，之后运行`lr`相当于运行`ls -lrth`（alias可以设置定制的命令）

## 13. 查找命令

which查找PATH环境变量下文件，通常用于明确当前环境执行的命令是哪一个

apropos根据关键词搜索文档中含有对应关键词的相关命令，可以拓展我们的命令知识面

## 14. 提交本地作业

sleep 100 &：后台运行命令

nohup sleep &：后台运行命令，并将控制台输出写入nohup.out文件中

jobs查看后台运行的作业

fg将后台运行的作业放在前台运行

