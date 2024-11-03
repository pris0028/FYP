# FYP: A Low-Power IoT-Enabled Bathroom Beacon for Elderly Safety
<img width="302" alt="image" src="https://github.com/user-attachments/assets/fca38e18-60ee-4415-9e6b-331190591d3f">


## Abstract
This report presents a secure and autonomous solution for detecting falls in bathroom environments, addressing privacy concerns associated with traditional video surveillance systems. The proposed approach integrates machine learning and artificial intelligence algorithms into edge processing devices, enabling real-time decision- making at the network's edge. The system utilizes advanced audio classification models to identify conscious occupants expressing fear when calling for help, complemented by obscured thermal imaging techniques to detect unconscious fallen individuals. The audio classifier employs a Deep Neural Network (DNN) architecture trained on the Toronto Emotional Speech Data Set (TESS), achieving an overall accuracy of 88.22% in recognizing emotions from vocalizations. The thermal image classifier analyses temperature differentials between image pixels, correctly identifying fallen postures with 96% recall and 38% precision when the optimal temperature threshold is applied. Extensive testing and evaluation of the system's performance are conducted, including the construction of a thermal image dataset and the incorporation of background bathroom noise into the audio classification model, reducing the fear detection accuracy to 72.73%. The report provides a comprehensive overview of the system's methodology, hardware and software architectures, data collection and training processes, and presents the results obtained from various test scenarios. Recommendations for future work and potential enhancements are also discussed, highlighting the system's potential for widespread adoption and its contribution to enhancing elderly safety in bathroom environments while prioritizing data privacy and security.

### Emotion Audio Recognition
*Confusion Matrix for Train Data*

<img width="328" alt="image" src="https://github.com/user-attachments/assets/57ba4aa2-50cf-407c-8625-c3742dc964fb">

*Confusion Matrix for Test Data*

<img width="333" alt="image" src="https://github.com/user-attachments/assets/9f33ce25-6c19-40e6-a4d8-f5cd86a6387a">

|Emotion |Train Accuracy|	Test Accuracy|
|---|---|---|
|Neutral	|0.9737	|0.9091|
|Happy	|0.9091	|0.9091|
|Sad	|0.7500	|0.8571|
|Angry	|0.8378	|0.9286|
|Fearful	|0.7941	|1.0000|
|Disgust	|0.9143	|1.0000|
|Surprised	|0.7647	|0.5714|
|Overall	|0.8491	|0.8822|

### Thermal Image Recognition
*Threshold of 1.2 times the temperature differential*
<img width="411" alt="image" src="https://github.com/user-attachments/assets/8b635900-5410-4f54-9531-77c72c3fcd0d">

|Classification|Precision|Recall|
|---|---|---|
|Fall|0.04|1.00|
|Not Fall|1.00|0.36|

*Threshold of 1 times the temperature differential*
<img width="411" alt="image" src="https://github.com/user-attachments/assets/31dcf1b7-10f4-4750-822a-ec2848d04282">

|Classification|Precision|Recall|
|---|---|---|
|Fall|0.38|1.00|
|Not Fall|1.00|0.96|

## Conclusion
In conclusion, while the proposed fall detection system demonstrates functionality and potential efficacy, the absence of real-world data presents a significant challenge in fully evaluating its performance. Despite successful implementation and validation in controlled settings, the lack of qualifiable data, particularly in the form of audio clips of real falls and thermal images of true falls, hinders the ability to comprehensively assess the system's capabilities. Moving forward, efforts should focus on obtaining relevant datasets that accurately represent the diversity of fall scenarios encountered in bathroom environments. Additionally, collaboration with healthcare professionals and caregivers could provide valuable insights and contribute to the refinement and validation of the system. Ultimately, addressing these challenges is crucial to ensuring the system's reliability and effectiveness in real-world applications, thereby enhancing the safety and well-being of individuals at risk of falls in bathroom settings.

*Terminal on Raspberry Pi to run the program*
<img width="665" alt="image" src="https://github.com/user-attachments/assets/cd7f3d90-d0f5-44b2-9f54-4d60bfeaf934">

*Telegram Call and Text Message in the event of a fall to the emergency contact*

<img width="305" alt="image" src="https://github.com/user-attachments/assets/c2665020-2e84-4d60-acd5-51710e7bccb7">


## Link to full report
https://dr.ntu.edu.sg/handle/10356/177045
