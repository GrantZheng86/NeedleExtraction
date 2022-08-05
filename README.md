# NeedleExtraction

The "main.py" program is used to find the curvature of a needle based on the markers.<br />

Usage:<br />
The main program taks two required image paths.<br /><br />
--calibration_dir_1 takes a string that points to a checkerboard ***image*** for camera 1 calibration. The checkerboard must have a pattern 8x4. The the checkerboard should also occupy 
as much area as possible to ensure a good calculation for camera calibration calculation.<br />
--calibration_dir_2 Same as the previous one. This is for camera 2<br/>
--line_segments The number of marker segments on the needle. This will be used to determine whether to save the regression
and points for the current frame or not<br />
<br />
The saved line regression parameters, line segments, and videos are saved in the same folder as the "Main" file. 
<br/>

To install required packages, navigate to the "NeedleExtraction" folder and run
```pip install -r requirements.txt```  <br/>

The recorded time includes milliseconds, seconds, minutes and hours. When the recorded times are displayed in Excel, the default formatting will not show
hour. To show the hour information, choose correct format option for each cell.
