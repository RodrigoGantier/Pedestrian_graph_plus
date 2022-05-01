# Pedestrian_graph_plus
This is a code repo for **[Pedestrian Graph +: A Fast Pedestrian Crossing Prediction Model based on Graph Convolutional Networks]
([T-ITS 2022](https://www.bilibili.com/video/BV1JB4y117Ho/)<br>

If you use any of this code, please cite the following publications:

```bibtex
@inproceedings{cadena2019pedestrian,
  title={Pedestrian graph: Pedestrian crossing prediction based on 2d pose estimation and graph convolutional networks},
  author={Cadena, Pablo Rodrigo Gantier and Yang, Ming and Qian, Yeqiang and Wang, Chunxiang},
  booktitle={2019 IEEE Intelligent Transportation Systems Conference (ITSC)},
  pages={2000--2005},
  year={2019},
  organization={IEEE}
}
```
## Set-up
install:
pytorch 1.8.o or above 
pytorch lightning 1.5.10 or above 

Also you can use docker
sudo docker pull nvcr.io/nvidia/l4t-base:r32.4.3
our code was tested on the jetson nano 4Gb
sudo run sh ./run_docker.sh

## Preliminary
- Download the linked material below
* Sample dataset for training and testing 
proprocess data ([data](https://pan.baidu.com/s/1GiBAR2voRvk15nI2wsKnUQ?pwd=1234)).
PIE data ([PIE](https://pan.baidu.com/s/1zKmftUUa96QXMnmOdc24Og?pwd=1234)).
JAAD data ([JAAD](https://pan.baidu.com/s/1EgOjuYXQuaSqr8m0jDdkUA?pwd=1234 )).

## Inference
test JAAD
```bash
python3 final_jaad_test.py --ckpt ./weigths/jaad-23-IVSFT/best.pth
```
test JAAD with 2D human keypoints
```bash
python3 final_jaad_test.py --ckpt ./weigths/jaad-23-IVSFT-h2d/best.pth
```
test PIE
```bash
python3 final_pie_test.py --ckpt ./weigths/jaad-23-IVSFT-h2d/best.pth
```

To train 
```bash
python3 pl_jaad_muster23_forecast.py --logdir ./weigths/jaad-23-IVSFT/
```

## License

MIT license
