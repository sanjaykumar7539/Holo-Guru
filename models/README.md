# Emotion model folder

Place a FER+ (or similar) facial-emotion ONNX model here as:

    models/emotion_ferplus.onnx

Expected input: grayscale 64x64 face crop, shape (1,1,64,64).
Output: 7-class logits in order:
    angry, disgust, fear, happy, neutral, sad, surprise

If this file is not present, core/emotion_detector.py still runs (face
detection works via OpenCV Haar cascade) but reports "neutral" for every
frame instead of a real prediction — the adaptive engine treats that as
"no signal" and does not change teaching behaviour. This keeps the app
fully functional even before you add a model file.

A ready-made ONNX FER+ model (~34MB) is available from the ONNX Model Zoo
under the name "emotion-ferplus" — download it once on a machine with
internet access and copy it into this folder on the Jetson.
