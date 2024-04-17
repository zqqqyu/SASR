# noise-free degradations with isotropic Gaussian blurs

python test.py --test_only \
               --dir_data='/home/zhangqianyu/SASR/' \
               --data_test='REALTEST' \
               --model='blindsr' \
               --scale='2' \
               --resume=600 \
               --blur_type='iso_gaussian' \
               --noise=0 \
               --sig=0

  
