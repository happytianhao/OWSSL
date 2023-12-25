# OWSSL

### Run for CIFAR10
```bash
python main.py --dataset_path data --num_workers 10 --batch_size 512 --epochs 200 --dataset CIFAR10 --n_class 10 --n_estimation 10 --n_labelled 5 --n_unlabelled 5
```

### Run for CIFAR100
```bash
python main.py --dataset_path data --num_workers 10 --batch_size 512 --epochs 200 --dataset CIFAR100 --n_class 100 --n_estimation 100 --n_labelled 50 --n_unlabelled 50
```
