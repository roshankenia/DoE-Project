import cv2
import numpy as np
vidcap = cv2.VideoCapture('Moving Pebbles - Ceramic Paint.MOV')
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('video has', str(frame_count), 'frames.')

rotations = [0, 45, 90, 135, 180, 225, 270, 315]
# sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

count = 0
while (vidcap.isOpened()):
    hasFrames, image = vidcap.read()
    if hasFrames:
        for rotation in rotations:
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
            result = cv2.warpAffine(
                image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            # result = cv2.filter2D(result, -1, sharpen_kernel)
            # save frame as JPG file
            cv2.imwrite("./ceramicimages/image"+str(count) +
                        "_"+str(rotation)+".jpg", result)
        count += 1
    else:
        break

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()
