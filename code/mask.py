import os
import cv2

def main():
    mask_path = "/___ Enter path ___/LRP/relevance/Final/Train/brir/"
    img_path = "/___ Enter path ___/LRP/mel/Final/Train/brir/"
    output_path = "/___ Enter path ___/LRP/output/Final/Train/brir/"
    filenames = [file for file in os.listdir(mask_path) if file.endswith(".png")]
    for file in sorted(filenames):
        img1 = cv2.imread(os.path.join(img_path, file))
        img = cv2.resize(img1, (1190, 1036))
        mask = cv2.imread(os.path.join(mask_path, file))

        output = cv2.addWeighted(img, 0.3, mask, 0.7, 0)
        # cv2.imshow("Overlay", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(output_path, file), output)

if __name__ == "__main__":
    main()
