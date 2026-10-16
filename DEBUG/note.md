注释关键词：按照 OpenPI 源码精注格式，给我完整代码和中文注释，所有参数分行注释，补全 Shape、英文全称、计算过程和具体示例，保证内容完整但尽量精简

1. 创建实例对象时：
    1) 普通 python 类：需要 __init__() 函数来实现实例属性的初始化
    2) 继承 nn.Module 的 类：Flax 的实例构造机制会识别声明的字段，在创建实例时将声明字段保存为实例属性，无须 __init__() 即可完成实例属性的初始化

2. 设置模块时：

3. 执行模型计算时

#######################################
# 步骤
#######################################

conda create -n openpi_debug python=3.11 -y
conda activate openpi_debug
python -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=120
uv sync --no-install-project

git remote add personal git@github.com:TonyBryant2084/PI_0.git
git clone git@github.com:TonyBryant2084/PI_0.git

# SSH
ssh-keygen -t ed25519 -C "tonybryant2084@gmail.com"
cat ~/.ssh/id_ed25519.pub



# 问题

1. 为什么克隆 AutoDL 实例后，可以直接通过 SSH 连接 GitHub？
原因：克隆实例保留了原实例的 SSH 私钥，而 GitHub 已保存对应的公钥。
~/.ssh/
├── id_ed25519       # SSH 私钥：用于身份验证
├── id_ed25519.pub   # SSH 公钥：对应公钥已添加到 GitHub
└── known_hosts      # 用于验证服务器身份
验证命令：
ssh -T git@github.com
结论： 新实例继承了原实例的 SSH 身份凭据（私钥...），因此无需重新配置密钥即可访问 GitHub。

2. 为什么 uv sync --no-install-project 执行得非常快？
原因：可能复用了原实例保留的 uv 缓存或已安装的虚拟环境。
设置清华 PyPI 镜像，可能加快依赖下载
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
设置 HTTP 超时时间为 120 秒（不直接提高下载速度）
export UV_HTTP_TIMEOUT=120
同步项目依赖，但不安装项目本身
uv sync --no-install-project
速度快的主要原因：
1. uv 缓存：复用已下载的依赖包，无需重复下载。
2. 已有 .venv：如果依赖已安装，uv 只需检查环境。
3. uv.lock：锁定依赖版本，减少重复解析。
4. uv 本身高效：支持并发下载和快速缓存安装。
查看缓存位置：
uv cache dir