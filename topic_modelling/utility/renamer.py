# -*- coding: utf-8 -*-
import os, sys, glob

for path in glob.glob('C:\\Temp\\Dedalus\\total-segmenterad-volumes\\total-segmenterade\\*.txt'):
    folder, filename = os.path.split(path)
    filename2 = filename.lower().replace('jj', '').replace(' #', '').replace('#', '').replace(' ', '_volume_')
    os.rename(os.path.join(folder, filename), os.path.join(folder, filename2))
    print('{} -> {}'.format(filename, filename2))
