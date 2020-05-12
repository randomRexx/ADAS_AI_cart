# ADAS op Zedboard/Ultra96

Project Onderzoek, in opdracht van [Vincent Claes](https://www.linkedin.com/in/vincentclaes/), te [PXL](https://www.pxl.be/). De bedoeling is om een ADAS te kunnen draaien op een Zedboard en een Ultra96, met een [IWR1642 radar](http://www.ti.com/tool/IWR1642BOOST) en [RealSense T265](https://www.intelrealsense.com/tracking-camera-t265/) camera/[RealSense D435](https://www.intelrealsense.com/depth-camera-d435/) camera.
De camera zal met een Vitis AI objecten detecteren, met een overlay van radar-waarden, gebruikmakend van ROS. Deze informatie wordt in een JSON-format doorgegeven in de UART.
