# Pedestrian_graph_plus
This is a code repo for **[Pedestrian Graph +: A Fast Pedestrian Crossing Prediction Model based on Graph Convolutional Networks]()**<br>


## Google colab
- Pedestrian Graph + Available via a [colab notebook](https://github.com/RodrigoGantier/Pedestrian_graph_plus/blob/main/Pedestrian_graph_plus.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg).


## wath Pedestrian Graph + on:

bilibili <br>

[![peaton](https://github.com/RodrigoGantier/Pedestrian_graph_plus/blob/main/docker/peaton.png)](https://www.bilibili.com/video/BV1JB4y117Ho/)<br>

Or on Youtube <br>

[![3d_ped](https://github.com/RodrigoGantier/Pedestrian_graph_plus/blob/main/docker/new_3d_ped.jpg)](https://www.youtube.com/watch?v=BZxf53VdyjU)<br>


## BibTeX
If you use any of this code, please cite the following publications:<br>

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
install:<br>
pytorch 1.8.0 or above <br>
pytorch lightning 1.5.10 or above <br>

Also you can use docker
```bash
sudo docker pull nvcr.io/nvidia/l4t-base:r32.4.3
```

```bash
sudo run sh ./run_docker.sh
```

our code was tested on the jetson nano 4Gb


## Preliminary
- Download the linked material below
Sample dataset for training and testing: <br>


pre-processed data baidu ([data](https://pan.baidu.com/s/1GiBAR2voRvk15nI2wsKnUQ?pwd=1234)).<br>
pre-processed data google dive ([data](https://drive.google.com/drive/folders/1I9TUDa7FpTrgSrf7_pivlB1O_pJ3U3u9?usp=sharing)).<br>

PIE data baidu ([PIE](https://pan.baidu.com/s/1zKmftUUa96QXMnmOdc24Og?pwd=1234)).<br>
PIE data google dive ([PIE](https://drive.google.com/drive/folders/1PNhyuiAhutkSg8xnc2uXvulm9P_b9De2?usp=sharing)).<br>

JAAD data baidu ([JAAD](https://pan.baidu.com/s/1EgOjuYXQuaSqr8m0jDdkUA?pwd=1234)).<br>
JAAD data google dive ([JAAD](https://drive.google.com/drive/folders/1HWAimRzwlvNBpUucULBb8BfCGht8R4dX?usp=sharing)).<br>

All three folders must be inside the Pedestrian_graph_plus folder <br>

## Inference
test JAAD
```bash
python3 final_jaad_test.py --ckpt ./weigths/jaad-23-IVSFT/best.pth
```
The following inference is made on a 4Gb jetson nano: <br>


![Pedestrian Graph +](https://github.com/RodrigoGantier/Pedestrian_graph_plus/blob/main/docker/jetson_nano.gif)<br>

This inference shows that **Pedestrian Graph +** is able to run on low-resource hardware, being efficient while maintaining high accuracy.<br>

![Pedestrian Graph +](https://github.com/RodrigoGantier/Pedestrian_graph_plus/blob/main/docker/gtx1060.gif)<br>

Inference time on jetson nano is 24ms, on the GTX 1060 (laptop) the inference time is 3 ms.<br>
 
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


## Qualitative Results

![3D_estimation](https://github.com/RodrigoGantier/Pedestrian_graph_plus/blob/main/docker/3D_estimation.png)<br>

![alpha_ped](https://github.com/RodrigoGantier/Pedestrian_graph_plus/blob/main/docker/alpha_ped.png)<br>

## License

MIT license
