# ADAS_AI_cart
Gemaakt door studenten van [de hogeschool PXL](https://www.pxl.be) op opdracht van [Vincent Claes](https://www.linkedin.com/in/vincentclaes/).

Het doel van dit project is een ADAS systeem ontwikkelen voor een golfkart autonoom te laten rijden.

Om objecten te kunnen detecteren wordt er gebruik gemaakt van een Pytorch resnet50-model en openCV die input krijgt van de [Intel T265 Realsense Tracking Camera](https://www.intelrealsense.com/tracking-camera-t265/) / [Intel D435 Realsense Camera]() en de [TI IWR1642 Radar](http://www.ti.com/tool/IWR1642BOOST).

Deze data wordt doorgestuurd naar de [Jetson-Nano van een meewerkende groep](https://github.com/KingAbad/Autonomous_Cart_2?fbclid=IwAR21YFBcbC4viqrMCfkstqgnDQ-sq7s7LPgTWAJHs7tx8XUIrGCixJqF12Q) via Json. 

Dit wordt geprogrameerd op de [Ultra96](http://zedboard.org/product/ultra96-v2-development-board) en/of het [zedboard](http://zedboard.org/product/zedboard) aan de hand van ROS2/ROS1 en/of U-boot met C-Kernels om een application te maken en worden achteraf vergeleken op effecientie.
## Branches
```
  Documentation                               algemene documentatie en instructies van installeren
  Radar                                       bestanden voor de TI IWR1642
  zedboard        
  zedboard_ros2_errors_solutions              Errors met het installeren van ROS2 en oplossingen
  zedboard-C-kernels                          C-test kernels voor Ubuntu 18.04 bionical
  zedboard-pytorch-resnet50-obj-detection     Resnet50 python programma met afbeeldingen
  zedboard-Uboot                              U-boot files voor op de zedbord
```
## Flowchart
<p align="center"><img src="flowchart.png"></p>

## Project Doelstellingen

## Gebruik gemaakt van

* [Python](https://www.python.org/) - gebruikte programeer taal voor het model.
* [OpenCV]() - gebruikte Library voor detectie visualisering.
* [ROS](https://www.ros.org/) - gebruikte tools-library om Camera en Radar samen te voegen (zedboard).
* [VITIS-AI](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html) - gebruikte ACAP systeem voor de Camera en Radar (Ultra96)

* [Intel RealSense Camera T265](https://www.intelrealsense.com/tracking-camera-t265/) - gebruikte camera (zedboard).
* [Intel D435 Camera]() - gebruikte camera (ultra96)
* [TI IWR1642](http://www.ti.com/tool/IWR1642BOOST) - gebruikte Radar.
* [zedboard](http://zedboard.org/product/zedboard) - gebruikte FPGA bord
* [Ultra96](http://zedboard.org/product/ultra96-v2-development-board) - gebruikte FPGA bord

## Auteurs
* *Vincent Claes*     -Docent / Product Owner- [LinkedIn](https://www.linkedin.com/in/vincentclaes/)
* *Bart Stukken*      -Docent / Scrum Master- [LinkedIn]()
* *Bart Gripsen*      -Student / Scrum Member- [LinkedIn]()
* *Dennis Merken*     -Student / Scrum Member- [LinkedIn]()
* *Jethro Pans*       -Student / Scrum Member- [LinkedIn](https://www.linkedin.com/in/jethro-pans-67518b1a3/)
* *Kris Teuwen*       -Student / Scrum Member- [LinkedIn]()
