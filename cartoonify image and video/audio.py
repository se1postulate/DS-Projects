# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:09:41 2021

@author: postulate-31
"""

import moviepy.editor as mp

clip = mp.VideoFileClip(r"m.mp4")
clip.audio.write_audiofile(r"audio.wav")