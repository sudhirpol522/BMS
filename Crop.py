# https://learnopencv.com/otsu-thresholding-with-opencv/ -> Othsu's Thresholding Algorithm
def crop(img, debug=False):
    if debug:
        fig, ax = plt.subplots(1,2, figsize=(30,8))
        ax[0].imshow(img)
        ax[0].set_title(f'original image, shape: {img.shape}', size=16)
    # In this we are going to use OTSu's Thresholding because if we are going to do it manually, during each tunnng result's will be diffrent.
    
        
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # it needs Grey image
    #https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html->cv2.RETR_LIST
    contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #it needs black and white image
    x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
    # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    for cnt in contours:
        
        x, y, w, h = cv2.boundingRect(cnt) #cv2.boundingRect takes coordinates(x,y) and gives the Approximate Box around that object
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    img_cropped = img[y_min:y_max, x_min:x_max]
    
    if debug:
        ax[1].imshow(img_cropped)
        ax[1].set_title(f'cropped image, shape: {img_cropped.shape}', size=16)
        plt.show()
    
    return img_cropped