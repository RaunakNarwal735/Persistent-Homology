from midiutil import MIDIFile
import random

mf = MIDIFile(2)

violin_track = 0
handpan_track = 1
time = 0

mf.addTempo(violin_track, time, 160)
mf.addTempo(handpan_track, time, 160)

mf.addTrackName(violin_track, time, "Intense Violin")
mf.addTrackName(handpan_track, time, "Handpan Rhythm")

mf.addProgramChange(violin_track, 0, 0, 40)   # Violin
mf.addProgramChange(handpan_track, 1, 0, 114)  # Handpan


scale = [57, 59, 60, 62, 64, 65, 67, 69]

t = 0
for i in range(120):
    base = random.choice(scale)
    arpeggio = [base, base+3, base+7, base+12]  
    
    for note in arpeggio:
        mf.addNote(
            violin_track,
            0,
            note,
            t,
            0.125,      
            random.randint(90, 120)
        )
        t += 0.125

t = 0
for i in range(240):
    pitch = random.choice([45, 47, 50, 52])  
    duration = random.choice([0.25, 0.5])

    mf.addNote(
        handpan_track,
        1,
        pitch,
        t,
        duration,
        random.randint(80, 110)
    )
    t += 0.25


with open("intense_violin_handpan.mid", "wb") as f:
    mf.writeFile(f)
