# MultiClassDA (TPAMI2020)
Code release for ["Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice"](https://arxiv.org/pdf/2002.08681.pdf), which is
an extension of our preliminary work of SymmNets [[Paper](https://zpascal.net/cvpr2019/Zhang_Domain-Symmetric_Networks_for_Adversarial_Domain_Adaptation_CVPR_2019_paper.pdf)] [[Code](https://github.com/YBZh/SymNets)]

Please refer to the "run_temp.sh" for the usage. 
All expeimental results are logged in the file of "./experiments"




## Included codes:
1. Codes of McDalNets -->./solver/McDalNet_solver.py
2. Codes of SymNets-V2
    1. For the Closed Set DA -->./solver/SymmNetsV2_solver.py
    2. For the Strongthened Closed Set DA -->./solver/SymmNetsV2SC_solver.py
    3. For the Partial DA  -->./solver/SSymmNetsV2Partial_solver.py
    4. For the Open Set DA  -->./solver/SymmNetsV2Open_solver.py


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
      journal=IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2020}
      publisher={IEEE}
    }

## Contact
If you have any problem about our code, feel free to contact
- zhang.yabin@mail.scut.edu.cn

or describe your problem in Issues. 
