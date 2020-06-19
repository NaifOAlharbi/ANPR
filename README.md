# ANPR
Automatic Number Plate Recognition system

The goal of this application is to detect the number plate of the cars and recognise the characters. It combines the image processing methods and the machine learning algorithms to limit the human interface.

The Program has two options: (images) and  (Videos)

1- Images:-
To run the program on images, please type  (-td=) before typing the path of the images or the name of the image(without any space between -td= and the path).
example:-

..\ANPRsystemData\ANPRsystem\x64\Debug> ANPRsystem.exe -td=test/*.JPG


2-Live-feed or Videos:-
In case of videos, please type(-tv=) before typing the path of the video or the name of the video.

example:-
..\ANPRsystemData\ANPRsystem\x64\Debug> ANPRsystem.exe -tv=Video1.MOV

For Live-feed, please type (-tv=0)
example:-
 ..\ANPRsystemData\ANPRsystem\x64\Debug> ANPRsystem.exe -tv=0

After typing the video name or the live feed, the program will ask the user to type a name for the output video (The recorded video), which will be saved in the Debug
Then, A window will be displayed to the user in order to adjust the bounding box for the Region Of Interest....



