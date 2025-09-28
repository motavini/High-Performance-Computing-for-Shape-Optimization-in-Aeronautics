SetFactory("OpenCASCADE");
Box(1) = {-15, -30, -20, 60, 60, 50};
Delete { Volume{1}; } // keep only box surface

SetFactory("Built-in");
Merge "bwb_remeshed.stl";

//+
Surface Loop(2) = {6, 1, 3, 5, 4, 2};
//+
Surface Loop(3) = {7};
//+
Volume(1) = {2,3};



