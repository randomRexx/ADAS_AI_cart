om de realsense tracking camera te runnen in een VM moet je op bepaalde dingen letten.
volgens de volgende URL:
https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md

is het aangeraden om VMWare te gebruiken i.p.v Oracle's VirtualBox, omwille van usb3.0 controller emulatie.

Ik heb Oracle VirtualBox gebruikt en de camera zal in het begin functioneren, alleen in toekomstig gebruik kan het zijn
dat de camera niet herkent wordt als de 
    "intel(R) Cooperation Intel(R) RealSense(TM) Tracking Camera T265 [FFFF]"
maar als de interne 'Movidus'.

bij het runnen van de demo "hello-realsense" komt de volgende error:
  RealSense error calling rs2_pipeline_start(pipe:0x....): no device connected

het is mogelijk om de camera terug herkent te laten worden.
de VM uitzetten en de usb drivers omzetten naar USB2.0
dan de demo opnieuw runnen. als het nog niet werkt, de camera uit/in steken in je [!!! USB3.0 !!!] aansluiting van je laptop/hub

tijdens de demo hoor je het geluid van (in/uisteken van een usb apparaat) en zou de movidus verwijnen en de intel camera verschijnen in je
USB apparaten.

/ to be continued /

