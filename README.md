A hybrid AI system that simulates autonomous drone decision making using Computer Vision, ML, and a C++ safety engine. The system processes telemetry data and visual input to decide whether a drone should continue flight, hold position, or return home. The project combines multiple components:

-Computer Vision perception  
-C++ safety rules engine  
-Machine Learning decision model  
-FastAPI backend  
-Real-time video streaming dashboard

The camera detects obstacles and updates the drone's perception in real time. It demonstrates a multi-layer autonomous drone decision pipeline similar to real robotics systems.
• Detect obstacles using OpenCV  
• Apply hard safety rules using C++  
• Use ML to make flight decisions  
• Stream camera data through a backend API  
• Simulate real drone telemetry

Stack:
-Python  
-FastAPI  
-OpenCV  
-Scikit-Learn  
-SHAP  
-C++ 

Tools:
-OpenCV – obstacle detection  
-FastAPI – backend API server  
-Uvicorn – ASGI server  
-Scikit-Learn – ML model  
-SHAP – decision explainability  
-Git – version control  

too see it working:
-Install dependencies: 
pip install fastapi uvicorn opencv-python scikit-learn shap pandas joblib
-Start the backend:
uvicorn backend.main:app --reload
-Open API docs:
http://127.0.0.1:8000/docs
-View camera feed:
http://127.0.0.1:8000/video_feed
