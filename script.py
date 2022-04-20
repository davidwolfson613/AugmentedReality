import cv2
import numpy as np

def get_corners_list(image):

    """
    Get image corner coordinates used in warping.
    """

    height, width = image.shape[:2]
    corners = [(0,0),(0,height - 1),(width - 1,0),(width - 1,height - 1)]

    return corners


def video_frame_generator(filename):

    # Open file with VideoCapture and set result to 'video'
    video = cv2.VideoCapture(filename)
    
    # yield frames, one at a time, until no more frames in video
    while video.isOpened():

        status,frame = video.read()

        if not status: # if there was an issue reading the frame
            break
        else:
            yield frame


    # Close video (release) and yield a 'None' value
    video.release()
    yield None


def find_markers(image, template=None):

    """
    Find corner markers in an image
    """

    # USING CORNER DETECTOR WITH KMEANS CLUSTERING
    img = np.copy(image)
    
    # need to convert image to grayscale then apply gaussian blur before passing the image to harris corner function
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(11,11),0)
    temp = cv2.cornerHarris(blur_img,7,7,0.05) # these parameters need a lot of tuning to get them to work

    # get corners
    corners = np.array(list(zip(*np.where(temp>0.1*temp.max())[::-1]))).astype(np.float32) # thresholding then getting coordinates using zip

    '''
    A lot of the coordinates in the corners array are very similar, so use kmeans clustering to find the best corners
    '''
    # define criteria and apply kmeans (criteria taken from OpenCV docs)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,_,centers=cv2.kmeans(corners,K=4,bestLabels=None,criteria=criteria,attempts=100,flags=cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint16(centers) # must use uint16 because some of the values are over 255

    # sort the centers
    centers = sorted(centers,key=lambda x:x[0]) # this sorts the points based on the x coordinate

    first_centers = sorted(centers[:2],key=lambda x:x[1]) # sort the first two centers based on the y coordinate
    second_centers = sorted(centers[2:],key=lambda x:x[1]) # sort the secodn two centers based on the y coordinate
    centers = np.array(first_centers + second_centers)

    # convert 2d array to list of tuples
    centersT = centers.T # tranpose corners
    out_list = list(zip(centersT[0],centersT[1]))

    return out_list


def project_images(imageA, imageB, homography):

    """
    This function projects imageA into the marked area of imageB.

    Uses find_markers method to find the corners.
    """

    # used this website as reference: https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function

    out_image = imageB.copy().astype(np.float32)
    src = imageA.copy().astype(np.float32)
    h,w,_ = imageB.shape

    # USE BACKWARD WARPING - TAKE PIXEL IN DEST IMAGE AND FIND IN SRC IMAGE

    # first find inverse homography matrix
    inv_h = np.linalg.inv(homography)

    # create coord map for dest image for backward warping
    y,x = np.indices((h,w))
    # print(y)
    coords = np.array([x.ravel(),y.ravel(),np.ones_like(x).ravel()])

    # do the warp
    # first matrix multiply inverse homography with dest image coordinates
    warp = np.matmul(inv_h,coords)

    # divide by scaling factor to get back x and y and reshape
    warp_x,warp_y = warp[:-1]/warp[-1]
    warp_x = warp_x.reshape(h,w).astype(np.float32)
    warp_y = warp_y.reshape(h,w).astype(np.float32)

    # do the remapping
    out_image = cv2.remap(src,warp_x,warp_y,cv2.INTER_LINEAR,dst=out_image,borderMode=cv2.BORDER_TRANSPARENT)
    out_image = np.uint8(out_image)

    # had black background instead of wall, tried using mask, didn't work
    # ind = np.where(out_image==(0,0,0)) # comment out this line for all parts other than part 5
    # print(ind)
    # exit()
    # out_image[ind] = src[ind] # comment out this line for all parts other than part 5

    return out_image


def find_homography(srcPoints, dstPoints):

    """
    This function performs a perspective transform on the provided points in the source and destination images.

    """
       
    # initialize the transform matrix to all zeros
    A = np.zeros((len(srcPoints)*2,9))
    
    # loop through source points and populate transform matrix
    for i in range(len(srcPoints)):
        for j in range(2):
            if j == 0:
                A[i*2+j] = np.array([srcPoints[i][0],srcPoints[i][1],1,0,0,0,-1*srcPoints[i][0]*dstPoints[i][0],-1*srcPoints[i][1]*dstPoints[i][0],-1*dstPoints[i][0]])
            else:
                A[i*2+j] = np.array([0,0,0,srcPoints[i][0],srcPoints[i][1],1,-1*srcPoints[i][0]*dstPoints[i][1],-1*srcPoints[i][1]*dstPoints[i][1],-1*dstPoints[i][1]])

    # invert the A matrix and multiply with homogenous vector, then reshape to 3x3
    _,_,h = np.linalg.svd(A)
    homography = h[-1,:]/h[-1,-1]

    return homography.reshape(3,3)


if __name__ == '__main__':

    filepath = r'.\wall.mp4'
    frames = video_frame_generator(filepath)

    frame = frames.__next__()
    h,w,_ = frame.shape
    # print(h,w)
    # cv2.imshow('hey',frame)
    # cv2.waitKey(0)
    # exit()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_vid = cv2.VideoWriter('output.mp4', fourcc, 40, (w,h))

    flag_img = cv2.imread(r'.\flag.png')
    src_points = get_corners_list(flag_img)

    num = 0
    while frame is not None:

        print('Processing frame {}'.format(num))
        markers = find_markers(frame)
        homography = find_homography(src_points,markers)
        out_img = project_images(flag_img,frame,homography)

        out_vid.write(out_img)
        num += 1
        frame = frames.__next__()

    out_vid.release()
