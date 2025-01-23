import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

class Padding():
    """
    For noise or silence padding.
    """
    def silence(self, file, branch, folder):
        """silence padding"""
        print(file)
        audio = AudioSegment.from_wav(os.path.join(branch, file))
        audio = audio.set_frame_rate(44100)
        pad_ms = 1.7 * 1000           #keeping length at 1.7s for now, later found the wall position of 1562 has info at 2s

        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/subset/silence/"
        if folder == 1:
            branch = os.path.join(path, "DH/")
        elif folder == 2:
            branch = os.path.join(path, "MTB/")
        else:
            branch = os.path.join(path, "SDM_KEMAR/")

        #assert pad_ms > len(audio), "Audio was longer that 1 second. Path: " + str(os.path.join(branch, file))
        if pad_ms > len(audio):
            silence = AudioSegment.silent(duration=pad_ms - len(audio))
            extended_audio = audio + silence  # Adding silence after the audio
            extended_audio.export(os.path.join(branch, file), format='wav')
        else:
            cut_audio = audio[:pad_ms]
            cut_audio.export(os.path.join(branch, file), format='wav')

    def noise(self, file, branch, folder):
        """noise padding"""
        print(file)
        audio = AudioSegment.from_wav(os.path.join(branch, file))
        audio = audio.set_frame_rate(44100)
        pad_ms = 1.7 * 1000           #keeping length at 1.7s for now, later found the wall position of 1562 has info at 2s

        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/Final/Test/noise/"
        if folder == 1:
            branch = os.path.join(path, "DH/")
        elif folder == 2:
            branch = os.path.join(path, "MTB/")
        else:
            branch = os.path.join(path, "SDM_KEMAR/")

        """
        #assert pad_ms > len(audio), "Audio was longer that 1 second. Path: " + str(os.path.join(branch, file))
        if pad_ms > len(audio):
            white_noise = WhiteNoise().to_audio_segment(duration=pad_ms - len(audio))
            a_audio = audio[-100:].dBFS
            a_noise = white_noise.dBFS
            a_change = a_audio - a_noise
            print(len(white_noise))
            final_noise = white_noise.apply_gain(a_change)
            print(len(final_noise))
            extended_audio = audio + final_noise    # Adding noise after the audio
            extended_audio.export(os.path.join(branch, file), format='wav')
        else:
            cut_audio = audio[:pad_ms]
            cut_audio.export(os.path.join(branch, file), format='wav')
        """
        if pad_ms > len(audio):
            extended_audio = audio
            while pad_ms > len(extended_audio):
                if (pad_ms - len(extended_audio)) > 50:
                    extended_audio = extended_audio + audio[-50:]
                else:
                    extend = pad_ms - len(extended_audio)
                    extended_audio = extended_audio + audio[-extend:]
            print(len(extended_audio))
            extended_audio.export(os.path.join(branch, file), format='wav')
        else:
            cut_audio = audio[:pad_ms]
            cut_audio.export(os.path.join(branch, file), format='wav')

    def get_files(self, folder, is_silent):
        """To open the files and send for operations"""
        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/subset/"
        if folder == 1:
            branch = os.path.join(path, "DH/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        elif folder == 2:
            branch = os.path.join(path, "MTB/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        else:
            branch = os.path.join(path, "SDM_KEMAR/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        print("Current folder is: ", branch)

        if (is_silent == 1):
            for file in sorted(filenames):
                self.silence(file, branch, folder)
        elif (is_silent == 2):
            for file in sorted(filenames):
                self.noise(file, branch, folder)
        print("Padding is completed")

def main():
    x = Padding()
    folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    while (folder != 1) and (folder != 2) and (folder != 3):
        if (folder == 1) or (folder == 2) or (folder == 3):
            break
        folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    is_silent = int(input("Enter 1 for silence, 2 for noise: "))
    while (is_silent != 1) and (is_silent != 2):
        if (is_silent == 1) or (is_silent == 2):
            break
        is_silent = int(input("Enter 1 for silence, 2 for noise: "))
    x.get_files(folder, is_silent)

if __name__ == "__main__":
    main()