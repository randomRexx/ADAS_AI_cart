# ADAS_AI_cart
in order for the 'object_detection.py' file to succesfully succeed. the SDK for the T265 and D435 need to be installed since it uses an extension library of these SDK's.

the program uses the default pre-trained city.pth model (the Path accociated to this file need to be edited in the object_detectrion.py file if used on a diffrent platform\ hardware)

using the D435 is optional and is used for extension purposes in case the TI radar malfunctions.(these are not implemented since the detection range falls to short for our used vehicle and might cause problems).

therefor the use of the D435 is made in a seperate python program able to be imported if needed.

