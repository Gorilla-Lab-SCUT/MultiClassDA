# MultiClassDA
Code release for ["Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice"](https://arxiv.org/pdf/2002.08681.pdf), which is
an extension of our preliminary work of SymmNets [[Paper](https://zpascal.net/cvpr2019/Zhang_Domain-Symmetric_Networks_for_Adversarial_Domain_Adaptation_CVPR_2019_paper.pdf)] [[Code](https://github.com/YBZh/SymNets)]

## Usage
For the convenience of potential users, we reimplement the project with the newest edition of the PyTorch (1.2.0).

We provide several script examples in the **run_temp.sh**, and the corresponding log file in the folder of **\experiments**.
You can start with these examples easily.  



## Code to be updated:
1. Code of McDalNets
    1. <del>For the Office-31, ImageCLEF, Office-Home, VisDA-2017 datasets (Finished)</del> 
    2. For the Digits dataset
2. Code of SymNets-V2
    1. For the Closed Set DA
        1. <del>Based on the ResNet (Finished)</del> 
        2. Based on the AlexNet
        3. For the Digits dataset
        4. Strengthened for Closed Set UDA
    2. For the Partial DA
        1. <del>Based on the ResNet (Finished)</del>
        2. Based on the AlexNet
    3. For the Open Set DA
        1. <del>Based on the ResNet (Finished)</del>


## Dataset
The structure of the dataset should be like

```
Office-31
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ ...
```


## Citation

    @inproceedings{zhang2019domain,
      title={Domain-symmetric networks for adversarial domain adaptation},
      author={Zhang, Yabin and Tang, Hui and Jia, Kui and Tan, Mingkui},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={5031--5040},
      year={2019}
    }
    @article{zhang2020unsupervised,
      title={Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice},
      author={Zhang, Yabin and Deng, Bin and Tang, Hui and Zhang, Lei and Jia, Kui},
      journal={arXiv preprint arXiv:2002.08681},
      year={2020}
    }

## Contact
If you have any problem about our code, feel free to contact
- zhang.yabin@mail.scut.edu.cn

or describe your problem in Issues. 
