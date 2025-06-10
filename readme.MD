# Driver Drowsiness Detector

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.5.3-orange.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.6.0-green.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time monitoring tool that detects driver drowsiness using computer vision and machine learning techniques. This project aims to enhance road safety by alerting drivers when signs of fatigue are detected.

## 🔧 Prerequisites

- **Python**: Version 3.8 or higher  

- **OpenCV**: Version 4.5.3

- **TensorFlow**: Version 2.6.0

- **Webcam**

## 🚀 Setup Instructions

### 1. **Clone & Navigate**
   ```bash
   git clone https://github.com/yourusername/DriverDrowsinessDetector.git
   cd DriverDrowsinessDetector
   ```

### 2. Set up and activate virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate (for Mac/Linux)
   .\venv\Scripts\activate.bat (for Windows)
   ```  

### 3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 4. **Environment Configuration (Optional)**
   Create a `.env` file in the root directory (if needed):
   ```env
   ALERT_THRESHOLD=your_alert_threshold_value
   ```

### 5. **Launch Application**
   ```bash
   python src/main.py
   ```

## 💡 Usage Guide

### Basic Commands
- **Run the Application**: Launch the main script to start monitoring.
- **Adjust Settings**: Modify alert thresholds in the `.env` file.

### Workflow
1. **Initialization**
   - Establishes webcam connection
   - Validates configuration settings

2. **Drowsiness Detection**
   - Analyzes video feed for signs of fatigue
   - Triggers alerts based on detection results

3. **User Notification**
   - Sends alerts to the driver when drowsiness is detected

## 🌟 Features

- 🔍 **Real-time Monitoring**
  - Utilizes webcam feed to analyze driver alertness
  - Detects eye closure and head position

- ⚠️ **Immediate Alerts**
  - Sends notifications to the driver when drowsiness is detected
  - Customizable alert settings

- 📊 **User-Friendly Interface**
  - Simple and intuitive design for easy interaction
  - Visual feedback on driver status

- ⚡ **Performance & Reliability**
  - Continuous monitoring with minimal latency
  - Robust error handling and recovery mechanisms

## 🔍 Troubleshooting

### Common Issues
- **Webcam Not Detected**: Ensure the webcam is connected and accessible.
- **Performance Issues**: Check system resources and close unnecessary applications.
- **Model Accuracy**: Adjust the alert threshold in the `.env` file for better sensitivity.

### Debug Tips
- Monitor console output for error messages
- Verify installation of required libraries
- Ensure proper permissions for webcam access

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

Your Name - [@affanabid](https://github.com/affanabid)

Project Link: [https://github.com/affanabid/DriverDrowsinessDetector](https://github.com/yourusername/DriverDrowsinessDetector)

---