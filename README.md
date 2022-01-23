# 1.Overview

This is the PaddlePaddle implementation of variational auto-encoder, applying on MNIST dataset.  
Currently, the following models are supported:  
✔️ VAE  
✔️ Conv-VAE

# 2.Usage  

### Train Model:  

```
CUDA_VISIBLE_DEVICES=0
sh run.sh
```
or
```
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --mode=convVAE \
    --result_dir=result \
    --save_dir=checkpoint \
    --batch_size=128 \
    --epoches=100 \
    --lr=1e-3 \
    --z_dim=20 \
    --input_dim=28*28 \
    --input_channels=1
```
you can also specify some customized options in ```train.py```  

### Generate Mnist Image:  

```
python generate.py \
    --mode=convVAE \
    --ckpt='' \ #指定模型参数文件路径
    --result_dir=generate_result
```

# 3.Result

Here are some visualization results:  

### ConvVAE:  
![random_sampled_90](https://user-images.githubusercontent.com/85541451/150643532-3c67fa59-1f10-4598-a498-d8f4fa58b5e3.png)
![random_sampled_80](https://user-images.githubusercontent.com/85541451/150643558-41789f95-17b9-4316-89c1-bd6767e80e4c.png)  

### VAE:  
![random_sampled_89](https://user-images.githubusercontent.com/85541451/150643567-9ecd198e-3ca8-4b6a-9708-5d3f553f9e6c.png)
![random_sampled_99](https://user-images.githubusercontent.com/85541451/150643575-f5423899-a3d2-433f-8936-0e9455d9e03b.png)
