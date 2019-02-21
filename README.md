# local3Ddescriptorlearning

This code implements the method of deep learning local descriptors for 3D surface shapes described in the following paper:

      --------------------------------------------------------------
      Hanyu Wang, Jianwei Guo, Dong-Ming Yan, Weize Quan, Xiaopeng Zhang. 
      Learning 3D Keypoint Descriptors for Non-Rigid Shape Matching. 
      ECCV 2018.
      --------------------------------------------------------------
      
Please consider citing the above paper if you use the code/program (or part of it). 

# License

This program is free software; you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation; either version 2 of 
the License, or (at your option) any later version. 

# Simple Usage

-Build cpp solution: this code is to generate geometry images. You can run this step in your local desktop.

	Modify CMakeLists ：
	add include_directories and link_directories for openmesh and matlab runtime
	Cmake
	
	Build solution:
	Modify config.ini for mesh_dir (directory of OFF models) gi_dir (directory of geometry images) and kpi_dir (directory of key points, you can skip it for dense matching) and other paras.
	
	MCC matlab "mcc -W cpplib:libcompcur -T link:lib compute_curvature.m"
	add libcompcur.dll to folder with GIGen.exe
	run "GIGen.exe config.ini", it will generate geometry images
	
-Python: this code is to train and test network. You should copy the geometry images generated in last step into the server.

	Train network:

		classify_gi_by_pidx_and_split.py -> tfr_gen.py -> train_softmax256.py -> train_mincv_perloss.py
		
-generate descriptor:
	
		descGen.py

# Contact
Should you have any questions, comments, or suggestions, please contact me at: 
gjianwei.000@gmail.com

Jianwei Guo: http://jianweiguo.net/

October, 2018

Copyright (C) 2018 
