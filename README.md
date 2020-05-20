# ERRORS and how to fix them

<details>
  <summary>ROS</summary>
  
  ## No moduled named lark
  
  * `roslib install "libasio-dev`
  * `python3 -m pip install -U lark-parser`
  * `apt install colcon-common-extensions`
  
  ## Colcon Build failed
  **undefined Refrence to UUID_generate@1.0/UUID_unparse_lower@1.0**
  <p align="center"><img src="./afbeeldingen (.png)/rviz_error.png"></p>
  
  This Error appeares when an another program isolates or overlays the main systems .SO files, causing a version diffrence. Programs like anaconda3 that create "sub_users/sub_accounts (like (base))" fall in this cathegory. The steps to fix this are a bit time consuming and there is probably a better solution for this, but these steps will fix it. (anaconda3 is used as an example)
  
  * check if anaconda3 appeares in the System Path, if so remove it from the PATH.
  check with `echo $PATH` in your terminal.
  to delete anaconda3 from the PATH do the following steps.
    * copy and save the current PATH (with anaconda3) in a text editor if anaconda3 is needed for future/other projects.
    * paste the copied PATH in a text editor and remove all paths where anaconda3 appeares.
    * copy the edited path and in the terminal type `PATH=<edited_path>`, where <edited_path> is the copied edited path and hit the enter key.
    
  Now anaconda3 can no longer interfere with future downloads, however the UUID error still exists. this is because the anaconda3 path is still inside of the "link.txt" and some "Cmake" files. these files exist in the following folders and need to be edited. 
 * `/<ros2_folder>/build/rviz_rendering/CMakeFiles/`
 * `/<ros2_folder>/build/rviz_rendering_tests/CMakeFiles/`
 * `/<ros2_folder>/build/rviz_common/CMakeFiles/`
 * `/<ros2_folder>/build/rviz_default_plugins/CMakeFiles/`
 * `/<ros2_folder>/build/rviz2/CMakeFiles/`
 
  where <ros2_folder> is the ros2 build folder, (where "colcon build" is used)
  
  for every "target.dir" without "autogen or action" in the name, the "link.txt" needs to be edited, But first open a second terminal and go to the following path:
 * `cd /usr/lib/x86_64-linux-gnu/`
 
 and search for any "libQT5.SO" file.
 
 A version number is writen in the name "ex. 5.9.5". Go back to the first terminal.
  
  * open the link.txt file in a text editor and use 'ctrl + f' to search for "anaconda3" (or "3" if the link.txt file is very big).
  
  * at the beginning of the link.txt file, remove the "/home/"user"/anaconda3/lib:".
  <p align="center"><img src="./afbeeldingen (.png)/link_txt_aanpassing_5.png"></p>
  
  * search further until the next anaconda3 paths appear
  <p align="center"><img src="./afbeeldingen (.png)/link_txt_aanpassing_1.png"></p>
  
  * change the path "/home/ "user" /anaconda3/lib" to `/usr/lib/x86_64-linux-gnu`.
  
  <p align="center"><img src="./afbeeldingen (.png)/link_txt_aanpassing_2.png"></p>
  
  * edit the .SO version to the version you see in the second terminal.
  
  <p align="center"><img src="./afbeeldingen (.png)/link_txt_aanpassing_3.png"></p>
  
  * to this for all remaining anaconda3 paths in the file, these can be positioned at random places, therefor use the 'ctrl + f' to find them all.
  
  <p align="center"><img src="./afbeeldingen (.png)/link_txt_aanpassing_4.png"></p>
  
  * save and exit the link.txt file and go to the next .dir file (without "autogen or target" in the name).
  
  in the `/ros2_build/build/rviz_defaulth_plugins/` folder exists a cmake_install.cmake file where anaconda3 is mentioned in the list of paths.
  
  * open the cmake_install.cmake file and search for the anaconda3 path.
  <p align="center"><img src="./afbeeldingen (.png)/cmake_install_cmake_aanpassing.png"></p>
  
  * remove this line and "save and exit".
  
The UUID error should now be fixed and "colcon build" can be executed succesfully.
</details>

<details>
  <summary>RealSense Camera</summary>
  
  ## Camera is not appearing in USB devices or can't be used ##
  This problem occures because the USB3.0 drivers from VM's are not allways compatible with the Realsense Camera.
  The following steps MAY fix this error, but they are not a garanteed/permanent fix.
  
  * check if annother Realsense camera of any type is connected, if so disconnect the device.
  * plug the camera in/out from your USB port (not on the camera, this may result a false start-up where the movidus is unnable to boot the realsense camera).
  * reboot linux
  * check if the USB3.0 drivers are enabled in the VM and if the Realsense camera is plugged into an USB3.0 input.
  * Full computer reboot
  * run any application with the realsense device with `sudo`.
  If these steps did not fix this problem then the usb3.0 drivers of the VM are not compatible to use the Realsense camera. 
  
</details>
<details>
  <summary>Radar</summary>
</details>
