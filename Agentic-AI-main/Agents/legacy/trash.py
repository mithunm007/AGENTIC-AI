
import soundfile as sf
import sounddevice as sd
import time

data, samplerate = sf.read("sounds/start.mp3")
# Play the audio
sd.play(data, samplerate)
sd.wait() 

time.sleep(1)


data, samplerate = sf.read("sounds/end.mp3")
# Play the audio
sd.play(data, samplerate)
sd.wait() 
