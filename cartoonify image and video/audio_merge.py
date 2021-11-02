# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:27:01 2021

@author: postulate-31
"""

from moviepy.editor import *
clip = VideoFileClip("m2.mp4")
audioclip = AudioFileClip(r"audio.wav")
videoclip = clip.set_audio(audioclip)
videoclip.ipython_display()
