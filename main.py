from lane import *
from moviepy.editor import VideoFileClip
import numpy as np

if __name__ == "__main__":

    demo = 1 # 1: image, 2 video

    if demo == 1:
        imagepath = r'E:\GeekyBeeAI\Internship project\driving-lane-line-detection\examples\test3.jpg'
        img = cv2.imread(imagepath)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        img_aug = process_frame(img)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
        ax2.imshow(img_aug)
        ax2.set_title('Augmented Image', fontsize=30)
        plt.show()

    else:
        video_output = r'E:\GeekyBeeAI\Internship project\driving-lane-line-detection\examples\project_video_augmented_final.mp4'
        clip1 = VideoFileClip(r"E:\GeekyBeeAI\Internship project\driving-lane-line-detection\examples\project_video.mp4")

        clip = clip1.fl_image(process_frame) #NOTE: it should be in BGR format
        clip.write_videofile(video_output, audio=False)

