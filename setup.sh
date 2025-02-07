set -ev
wget https://github.com/deepmodeling/deepmd-kit/releases/download/v3.0.1/deepmd-kit-3.0.1-cuda126-Linux-x86_64.sh.0
wget https://mirror.nju.edu.cn/github-release/deepmodeling/deepmd-kit/v3.0.1/deepmd-kit-3.0.1-cuda126-Linux-x86_64.sh.1
cat deepmd-kit-3.0.1-cuda126-Linux-x86_64.sh.0 deepmd-kit-3.0.1-cuda126-Linux-x86_64.sh.1 > deepmd-kit-3.0.1-cuda126-Linux-x86_64.sh
bash deepmd-kit-3.0.1-cuda126-Linux-x86_64.sh -b
source ~/deepmd-kit/bin/activate
conda install jax==0.4.34 -y -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
pip install nvidia-cuda-nvcc-cu12 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
rm -f deepmd-kit-3.0.1-cuda126-Linux-x86_64.sh*
git clone https://gitee.com/njzjz/benchmark-dpv3
cd benchmark-dpv3
python make_model.py
python run_sim.py
cat benchmark.out | tee fp64.out
sed -i 's/float64/float32/g' inputs/*.json
python make_model.py
python run_sim.py
cat benchmark.out | tee fp32.out
