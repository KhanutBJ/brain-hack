from gtts import gTTS
import os

def TTS(text, language='en'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language)

    # Save the speech as an audio file
    tts.save("output.mp3")

    # Play the saved audio file
    os.system("afplay output.mp3")
    
    # delete temp file
    os.remove("output.mp3")

