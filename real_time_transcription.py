#! python3.7

import argparse
import os
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
import numpy as np
import speech_recognition
import torch
from faster_whisper import WhisperModel
from sys import platform
from threading import Lock
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def _helper_setup_microphone(args):
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(speech_recognition.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(speech_recognition.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = speech_recognition.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = speech_recognition.Microphone(sample_rate=16000)
    return source

def save_transcription(transcription):
    print(rf'Writing transcription to: C:\Users\admin\Documents\ASR\real-time transcription\whisper_real_time\transcription.txt')
    path = r"C:\Users\admin\Documents\ASR\real-time transcription\whisper_real_time\transcription.txt"
    try:
        with open(path, "w") as file:
            for item in transcription:
                file.write(f"{item}\n")
    except Exception as e:
        print(f"Error writing transcription: {e}")
        print(f"Transcription shown on console: {transcription}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distil-large-v3", help="Model to use",
                        choices=["tiny", "tiny.en", "base", "base.en", "small", 
                                 "small.en", "distil-small.en", "medium", "medium.en", 
                                 "distil-medium.en", "large-v3", "distil-large-v3"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=60,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=0.5,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                  "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    queue_lock = Lock()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = speech_recognition.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    source = _helper_setup_microphone(args)
    if not source:
        print("Error with connecting to microphone.")
        return

    #optimizing transcription time using faster-whisper
    try:
        audio_model = WhisperModel(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:speech_recognition.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        with queue_lock:
            data = audio.get_raw_data()
            data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Model loaded.\n")
    def process_audio_chunk(audio_np):
        result = audio_model.transcribe(audio_np, language="en", beam_size=5, best_of=5, temperature=0.4)[0]
        return "".join(segment.text for segment in result)
        
    i = 0
    transcription = []
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                    
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                with queue_lock:
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()
                
                # Necessary conversion
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                partial_text = process_audio_chunk(audio_np)
                transcription.append(partial_text)

                if i == 0:
                    print("TRANSCRIPTION HAS STARTED")
                    print("-------------------------")
                print(transcription[i])
                i = i+1

            else:
                sleep(0.05)
        except KeyboardInterrupt:
            break
        
    print('\n\nENDING TRANSCRIPTION')
    save_transcription(transcription)


if __name__ == "__main__":
    main()
