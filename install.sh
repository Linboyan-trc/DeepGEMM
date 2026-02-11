# Change current directory into project root
# 1. original = 当前目录/DeepGEMM
# 1.1 script_dir = insatll.sh上级目录也是/DeepGEMM
# 1.2 然后还是进入到当前目录/DeepGEMM
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and install
# 2. 删除打包文件
# 2.1 build，构建临时文件
# 2.2 dist，生成的wheel包
# 2.3 *.egg-info，包的元信息
rm -rf build dist
rm -rf *.egg-info

# 3. 构建二进制分发包wheel，最终在/DeepGEMM/dist下生成.whl文件
# 3.1 找到刚才生成的.whl文件，用pip安装
# 3.2 回到original = 当前目录/DeepGEMM
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
cd "$original_dir"
