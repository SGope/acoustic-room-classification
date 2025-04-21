import os
from pydub import AudioSegment

class Cut():
    def cut_audio(self, file, branch):
        """Cutting audio"""
        print(file)
        audio = AudioSegment.from_wav(os.path.join(branch, file))
        audio = audio.set_frame_rate(44100)
        pad_ms = 1.7 * 1000  # keeping length at 1.7s

        path = "/___ Enter path ___/"

        if pad_ms < len(audio):
            cut_audio = audio[:pad_ms]
            print(len(cut_audio))
            cut_audio.export(os.path.join(path, file), format='wav')

    def get_files(self):
        """To get files for processing"""
        path = "/___ Enter path ___/"
        filenames = [file for file in os.listdir(path) if file.endswith(".wav")]
        print("Current folder is: ", path)

        for file in sorted(filenames):
            self.cut_audio(file, path)
        print("Plots have been saved")

def main():
    x = Cut()
    x.get_files()

if __name__ == "__main__":
    main()
