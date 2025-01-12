cd /wanglishuang/working_space/evogp/;
export PYTHONPATH="./src";
/root/miniconda3/envs/evogp_env/bin/python setup.py build_ext --inplace;
/root/miniconda3/envs/evogp_env/bin/python ./experiment/constant_exp2.py;
