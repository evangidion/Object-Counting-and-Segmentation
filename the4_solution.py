# Andaç Berkay Seval 2235521

import cv2
import os
import numpy as np
from skimage import io, img_as_ubyte
from skimage.morphology import square, disk, star
from skimage.morphology import binary_erosion, binary_opening, binary_dilation, binary_closing, thin
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import networkx as nx
from skimage.transform import rescale
import skimage.measure
from skimage import data, segmentation, color

INPUT_PATH = "./THE4_Images/"
OUTPUT_PATH = "./Outputs/" 

def object_counting(n):
    ### for image 1
    if n == 1:
        img = io.imread(INPUT_PATH + "A1.png", as_gray = True)
        img = img_as_ubyte(img)
        # thresh = threshold_otsu(img)
        # print(thresh)
        # binary = img > thresh
        ### binarize image with a threshold
        binary = img > 43
        binary = img_as_ubyte(binary)
        ### create a structuring element to apply opening to image
        b = square(5)
        opening = binary_opening(binary, b)
        opening = img_as_ubyte(opening)
        ### apply thinning to opened image
        thinned = thin(opening, max_iter=5)
        thinned = img_as_ubyte(thinned)
        ### apply erosion to thinned image
        erose = binary_erosion(thinned, b)
        erose = img_as_ubyte(erose)
        ### apply opening to erosed image
        opening = binary_opening(erose, b)
        opening = img_as_ubyte(opening)
        ### apply thinning to opened image
        thinned = thin(opening, max_iter=20)
        thinned = img_as_ubyte(thinned)
        ### create a new structuring element to apply erosion to thinned image
        b2 = disk(3)
        erose = binary_erosion(thinned, b2)
        erose = img_as_ubyte(erose)
        ### delete unnecesssary connected components to reduce noise
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(erose, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 1500:  
                result[labels == i + 1] = 255
        io.imsave(OUTPUT_PATH + "A1.png", result)
        labeled_image, count = skimage.measure.label(result, connectivity=2, return_num=True)
        print("The number of flowers in image A1 is " + str(count))

    ### for image 2
    elif n == 2:
        img = io.imread(INPUT_PATH + "A2.png", as_gray = True)
        img = img_as_ubyte(img)
        # thresh = threshold_otsu(img)
        # print(thresh)
        # binary = img > thresh
        ### binarize image with a threshold
        binary = img > 80
        binary = img_as_ubyte(binary)
        ### create a structuring element to apply opening to image
        b = square(5)
        opening = binary_opening(binary, b)
        opening = img_as_ubyte(opening)
        ### create a new structuring element to apply closing to opened image
        b2 = square(20)
        closing = binary_closing(opening, b2)
        closing = img_as_ubyte(closing)
        ### delete unnecesssary connected components to reduce noise
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 200000:  
                result[labels == i + 1] = 255
        io.imsave(OUTPUT_PATH + "A2.png", result)
        labeled_image, count = skimage.measure.label(result, connectivity=2, return_num=True)
        print("The number of flowers in image A2 is " + str(count))

    ### for image 3
    else:
        img = io.imread(INPUT_PATH + "A3.png", as_gray = True)
        img = img_as_ubyte(img)
        # thresh = threshold_otsu(img)
        # print(thresh)
        # binary = img > thresh
        ### binarize image with a threshold
        binary = img > 130
        binary = img_as_ubyte(binary)
        ### create a structuring element to apply opening to image
        b = square(3)
        opening = binary_opening(binary, b)
        opening = img_as_ubyte(opening)
        ### delete unnecesssary connected components to reduce noise
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 50000:  
                result[labels == i + 1] = 255
        ### create a new structuring element to apply erosion to image
        b3 = disk(40)
        result = binary_erosion(result, b3)
        result = img_as_ubyte(result)
        ### delete unnecesssary connected components to reduce noise
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(result, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 100:  
                result[labels == i + 1] = 255
        io.imsave(OUTPUT_PATH + "A3.png", result)
        labeled_image, count = skimage.measure.label(result, connectivity=2, return_num=True)
        print("The number of flowers in image A3 is " + str(count))


def clustering(n, t):
    ### for image 1
    if n == 1:
        image = io.imread(INPUT_PATH + "B1.jpg")
        ### for mean shift segmentation
        if t == "meanshift":
            ### rescale the image for mean shift segmentation in order to process the algorithm in a logical amount of time
            img = rescale(image, 0.25, anti_aliasing=False, multichannel=True)
            io.imsave(OUTPUT_PATH + "B1_algorithm_meanshift_original.png", img)
            shape = img.shape
            flatimg = np.reshape(img, [-1, 3])
            ### for image 1, first parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 75
            bandwidth1 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=75)
            ### apply mean shift segmentation to image 1 with the first parameter set
            ms = MeanShift(bandwidth = bandwidth1, bin_seeding=True)
            ms.fit(flatimg)
            labels=ms.labels_
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels)
            labels = np.reshape(labels, (1008,756))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented = color.label2rgb(labels, img, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_1_segmentationmap.png", segmented)

            ### boundary overlay
            boundary = mark_boundaries(img, labels, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            g = graph.rag_mean_color(img, labels, mode='similarity')
            nx.draw(g)
            plt.savefig(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 1, second parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 50
            bandwidth2 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=50)
            ### apply mean shift segmentation to image 1 with the second parameter set
            ms2 = MeanShift(bandwidth = bandwidth2, bin_seeding=True)
            ms2.fit(flatimg)
            labels2=ms2.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels2)
            labels2 = np.reshape(labels2, (1008,756))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented2 = color.label2rgb(labels2, img, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_2_segmentationmap.png", segmented2)

            ### boundary overlay
            boundary2 = mark_boundaries(img, labels2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            g2 = graph.rag_mean_color(img, labels2, mode='similarity')
            nx.draw(g2)
            plt.savefig(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 1, third parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 25
            bandwidth3 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=25)
            ### apply mean shift segmentation to image 1 with the third parameter set
            ms3 = MeanShift(bandwidth = bandwidth3, bin_seeding=True)
            ms3.fit(flatimg)
            labels3=ms3.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels3)
            labels3 = np.reshape(labels3, (1008,756))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented3 = color.label2rgb(labels3, img, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_3_segmentationmap.png", segmented3)

            ### boundary overlay
            boundary3 = mark_boundaries(img, labels3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            g3 = graph.rag_mean_color(img, labels3, mode='similarity')
            nx.draw(g3)
            plt.savefig(OUTPUT_PATH + "B1_algorithm_meanshift_parameterset_3_rag.png")  
            plt.clf()      

        ### for n-cut segmentation
        elif t == "ncut":
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_original.png", image)
            ### for image 1, first parameter set is in labels of the n-cut algorithm: compactness = 10 and number of segments = 300
            segments = segmentation.slic(image, compactness=10, n_segments=300)
            g = graph.rag_mean_color(image, segments, mode='similarity')
            ### apply n-cut segmentation to image 1 with the first parameter set 
            graph_cut_segments = graph.cut_normalized(segments, g)
            # labels_unique = np.unique(graph_cut_segments)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented = color.label2rgb(graph_cut_segments, image, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_1_segmentationmap.png", segmented)
            
            ### boundary overlay
            boundary = segmentation.mark_boundaries(image, graph_cut_segments, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            rag = graph.rag_mean_color(image, graph_cut_segments, mode='similarity')
            nx.draw(rag)
            plt.savefig(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 1, second parameter set is in labels of the n-cut algorithm: compactness = 10 and number of segments = 500
            segments2 = segmentation.slic(image, compactness=10, n_segments=500)
            g2 = graph.rag_mean_color(image, segments2, mode='similarity')
            ### apply n-cut segmentation to image 1 with the second parameter set 
            graph_cut_segments2 = graph.cut_normalized(segments2, g2)
            # labels_unique = np.unique(graph_cut_segments2)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented2 = color.label2rgb(graph_cut_segments2, image, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_2_segmentationmap.png", segmented2)
            
            ### boundary overlay
            boundary2 = segmentation.mark_boundaries(image, graph_cut_segments2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            rag2 = graph.rag_mean_color(image, graph_cut_segments2, mode='similarity')
            nx.draw(rag2)
            plt.savefig(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 1, third parameter set is in labels of the n-cut algorithm: compactness = 30 and number of segments = 500
            segments3 = segmentation.slic(image, compactness=30, n_segments=500)
            g3 = graph.rag_mean_color(image, segments3, mode='similarity')
            ### apply n-cut segmentation to image 1 with the third parameter set 
            graph_cut_segments3 = graph.cut_normalized(segments3, g3)
            # labels_unique = np.unique(graph_cut_segments3)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented3 = color.label2rgb(graph_cut_segments3, image, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_3_segmentationmap.png", segmented3)
            
            ### boundary overlay
            boundary3 = segmentation.mark_boundaries(image, graph_cut_segments3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            rag3 = graph.rag_mean_color(image, graph_cut_segments3, mode='similarity')
            nx.draw(rag3)
            plt.savefig(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_3_rag.png")
            plt.clf()

    ### for image 2
    elif n == 2:
        image = io.imread(INPUT_PATH + "B2.jpg")
        ### for mean shift segmentation
        if t == "meanshift":
            ### rescale the image for mean shift segmentation in order to process the algorithm in a logical amount of time
            img = rescale(image, 0.25, anti_aliasing=False, multichannel=True)
            io.imsave(OUTPUT_PATH + "B2_algorithm_meanshift_original.png", img)
            shape = img.shape
            flatimg = np.reshape(img, [-1, 3])
            ### for image 2, first parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 300
            bandwidth1 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=300)
            ### apply mean shift segmentation to image 2 with the first parameter set
            ms = MeanShift(bandwidth = bandwidth1, bin_seeding=True)
            ms.fit(flatimg)
            labels=ms.labels_
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels)
            labels = np.reshape(labels, (1008,756))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented = color.label2rgb(labels, img, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_1_segmentationmap.png", segmented)

            ### boundary overlay
            boundary = mark_boundaries(img, labels, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            g = graph.rag_mean_color(img, labels, mode='similarity')
            nx.draw(g)
            plt.savefig(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 2, second parameter set is in bandwith of the mean shift algorithm: quantile = 0.2 and number of samples = 700
            bandwidth2 = estimate_bandwidth(flatimg, quantile=0.2, n_samples=700)
            ### apply mean shift segmentation to image 2 with the second parameter set
            ms2 = MeanShift(bandwidth = bandwidth2, bin_seeding=True)
            ms2.fit(flatimg)
            labels2=ms2.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels2)
            labels2 = np.reshape(labels2, (1008,756))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented2 = color.label2rgb(labels2, img, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_2_segmentationmap.png", segmented2)

            ### boundary overlay
            boundary2 = mark_boundaries(img, labels2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            g2 = graph.rag_mean_color(img, labels2, mode='similarity')
            nx.draw(g2)
            plt.savefig(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 2, third parameter set is in bandwith of the mean shift algorithm: quantile = 0.3 and number of samples = 100
            bandwidth3 = estimate_bandwidth(flatimg, quantile=0.3, n_samples=100)
            ### apply mean shift segmentation to image 2 with the third parameter set
            ms3 = MeanShift(bandwidth = bandwidth3, bin_seeding=True)
            ms3.fit(flatimg)
            labels3=ms3.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels3)
            labels3 = np.reshape(labels3, (1008,756))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented3 = color.label2rgb(labels3, img, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_3_segmentationmap.png", segmented3)

            ### boundary overlay
            boundary3 = mark_boundaries(img, labels3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            g3 = graph.rag_mean_color(img, labels3, mode='similarity')
            nx.draw(g3)
            plt.savefig(OUTPUT_PATH + "B2_algorithm_meanshift_parameterset_3_rag.png") 
            plt.clf()

        ### for n-cut segmentation
        elif t == "ncut":
            io.imsave(OUTPUT_PATH + "B2_algorithm_ncut_original.png", image)
            ### for image 2, first parameter set is in labels of the n-cut algorithm: compactness = 1 and number of segments = 300
            segments = segmentation.slic(image, compactness=1, n_segments=300)
            g = graph.rag_mean_color(image, segments, mode='similarity')
            ### apply n-cut segmentation to image 2 with the first parameter set 
            graph_cut_segments = graph.cut_normalized(segments, g)
            # labels_unique = np.unique(graph_cut_segments)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented = color.label2rgb(graph_cut_segments, image, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_1_segmentationmap.png", segmented)
            
            ### boundary overlay
            boundary = segmentation.mark_boundaries(image, graph_cut_segments, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            rag = graph.rag_mean_color(image, graph_cut_segments, mode='similarity')
            nx.draw(rag)
            plt.savefig(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 2, second parameter set is in labels of the n-cut algorithm: compactness = 10 and number of segments = 300
            segments2 = segmentation.slic(image, compactness=10, n_segments=300)
            g2 = graph.rag_mean_color(image, segments2, mode='similarity')
            ### apply n-cut segmentation to image 2 with the second parameter set 
            graph_cut_segments2 = graph.cut_normalized(segments2, g2)
            # labels_unique = np.unique(graph_cut_segments2)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented2 = color.label2rgb(graph_cut_segments2, image, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B1_algorithm_ncut_parameterset_2_segmentationmap.png", segmented2)
            
            ### boundary overlay
            boundary2 = segmentation.mark_boundaries(image, graph_cut_segments2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            rag2 = graph.rag_mean_color(image, graph_cut_segments2, mode='similarity')
            nx.draw(rag2)
            plt.savefig(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 2, third parameter set is in labels of the n-cut algorithm: compactness = 10 and number of segments = 500
            segments3 = segmentation.slic(image, compactness=10, n_segments=500)
            g3 = graph.rag_mean_color(image, segments3, mode='similarity')
            ### apply n-cut segmentation to image 2 with the third parameter set 
            graph_cut_segments3 = graph.cut_normalized(segments3, g3)
            # labels_unique = np.unique(graph_cut_segments3)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented3 = color.label2rgb(graph_cut_segments3, image, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_3_segmentationmap.png", segmented3)
            
            ### boundary overlay
            boundary3 = segmentation.mark_boundaries(image, graph_cut_segments3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            rag3 = graph.rag_mean_color(image, graph_cut_segments3, mode='similarity')
            nx.draw(rag3)
            plt.savefig(OUTPUT_PATH + "B2_algorithm_ncut_parameterset_3_rag.png") 
            plt.clf()         

    ### for image 3
    elif n == 3:
        img = io.imread(INPUT_PATH + "B3.jpg")
        ### for mean shift segmentation
        if t == "meanshift":
            io.imsave(OUTPUT_PATH + "B3_algorithm_meanshift_original.png", img)
            shape = img.shape
            flatimg = np.reshape(img, [-1, 3])
            ### for image 3, first parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 300
            bandwidth1 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=300)
            ### apply mean shift segmentation to image 3 with the first parameter set
            ms = MeanShift(bandwidth = bandwidth1, bin_seeding=True)
            ms.fit(flatimg)
            labels=ms.labels_
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels)
            labels = np.reshape(labels, (443,666))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented = color.label2rgb(labels, img, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_1_segmentationmap.png", segmented)

            ### boundary overlay
            boundary = mark_boundaries(img, labels, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            g = graph.rag_mean_color(img, labels, mode='similarity')
            nx.draw(g)
            plt.savefig(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 3, second parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 700
            bandwidth2 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=700)
            ### apply mean shift segmentation to image 3 with the second parameter set
            ms2 = MeanShift(bandwidth = bandwidth2, bin_seeding=True)
            ms2.fit(flatimg)
            labels2=ms2.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels2)
            labels2 = np.reshape(labels2, (443,666))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented2 = color.label2rgb(labels2, img, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_2_segmentationmap.png", segmented2)

            ### boundary overlay
            boundary2 = mark_boundaries(img, labels2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            g2 = graph.rag_mean_color(img, labels2, mode='similarity')
            nx.draw(g2)
            plt.savefig(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 3, third parameter set is in bandwith of the mean shift algorithm: quantile = 0.2 and number of samples = 100
            bandwidth3 = estimate_bandwidth(flatimg, quantile=0.2, n_samples=100)
            ### apply mean shift segmentation to image 3 with the third parameter set
            ms3 = MeanShift(bandwidth = bandwidth3, bin_seeding=True)
            ms3.fit(flatimg)
            labels3=ms3.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels3)
            labels3 = np.reshape(labels3, (443,666))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented3 = color.label2rgb(labels3, img, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_3_segmentationmap.png", segmented3)

            ### boundary overlay
            boundary3 = mark_boundaries(img, labels3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            g3 = graph.rag_mean_color(img, labels3, mode='similarity')
            nx.draw(g3)
            plt.savefig(OUTPUT_PATH + "B3_algorithm_meanshift_parameterset_3_rag.png")   
            plt.clf()    

        ### for n-cut segmentation
        elif t == "ncut":
            io.imsave(OUTPUT_PATH + "B3_algorithm_ncut_original.png", img)
            ### for image 3, first parameter set is in labels of the n-cut algorithm: compactness = 1 and number of segments = 300
            segments = segmentation.slic(img, compactness=1, n_segments=300)
            g = graph.rag_mean_color(img, segments, mode='similarity')
            ### apply n-cut segmentation to image 3 with the first parameter set 
            graph_cut_segments = graph.cut_normalized(segments, g)
            # labels_unique = np.unique(graph_cut_segments)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented = color.label2rgb(graph_cut_segments, img, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_1_segmentationmap.png", segmented)
            
            ### boundary overlay
            boundary = segmentation.mark_boundaries(img, graph_cut_segments, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            rag = graph.rag_mean_color(img, graph_cut_segments, mode='similarity')
            nx.draw(rag)
            plt.savefig(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 3, second parameter set is in labels of the n-cut algorithm: compactness = 1 and number of segments = 200
            segments2 = segmentation.slic(img, compactness=1, n_segments=200)
            g2 = graph.rag_mean_color(img, segments2, mode='similarity')
            ### apply n-cut segmentation to image 3 with the second parameter set 
            graph_cut_segments2 = graph.cut_normalized(segments2, g2)
            # labels_unique = np.unique(graph_cut_segments2)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented2 = color.label2rgb(graph_cut_segments2, img, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_2_segmentationmap.png", segmented2)
            
            ### boundary overlay
            boundary2 = segmentation.mark_boundaries(img, graph_cut_segments2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            rag2 = graph.rag_mean_color(img, graph_cut_segments2, mode='similarity')
            nx.draw(rag2)
            plt.savefig(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 3, third parameter set is in labels of the n-cut algorithm: compactness = 10 and number of segments = 500
            segments3 = segmentation.slic(img, compactness=10, n_segments=500)
            g3 = graph.rag_mean_color(img, segments3, mode='similarity')
            ### apply n-cut segmentation to image 3 with the third parameter set 
            graph_cut_segments3 = graph.cut_normalized(segments3, g3)
            # labels_unique = np.unique(graph_cut_segments3)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented3 = color.label2rgb(graph_cut_segments3, img, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_3_segmentationmap.png", segmented3)
            
            ### boundary overlay
            boundary3 = segmentation.mark_boundaries(img, graph_cut_segments3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            rag3 = graph.rag_mean_color(img, graph_cut_segments3, mode='similarity')
            nx.draw(rag3)
            plt.savefig(OUTPUT_PATH + "B3_algorithm_ncut_parameterset_3_rag.png")
            plt.clf()

    ### for image 4
    else:
        image = io.imread(INPUT_PATH + "B4.jpg")
        ### for mean shift segmentation
        if t == "meanshift":
            ### rescale the image for mean shift segmentation in order to process the algorithm in a logical amount of time
            img = rescale(image, 0.25, anti_aliasing=False, multichannel=True)
            io.imsave(OUTPUT_PATH + "B4_algorithm_meanshift_original.png", img)
            shape = img.shape
            flatimg = np.reshape(img, [-1, 3])
            ### for image 4, first parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 300
            bandwidth1 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=300)
            ### apply mean shift segmentation to image 4 with the first parameter set
            ms = MeanShift(bandwidth = bandwidth1, bin_seeding=True)
            ms.fit(flatimg)
            labels=ms.labels_
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels)
            labels = np.reshape(labels, (756,1008))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented = color.label2rgb(labels, img, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_1_segmentationmap.png", segmented)

            ### boundary overlay
            boundary = mark_boundaries(img, labels, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            g = graph.rag_mean_color(img, labels, mode='similarity')
            nx.draw(g)
            plt.savefig(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 4, second parameter set is in bandwith of the mean shift algorithm: quantile = 0.2 and number of samples = 700
            bandwidth2 = estimate_bandwidth(flatimg, quantile=0.2, n_samples=700)
            ### apply mean shift segmentation to image 4 with the second parameter set
            ms2 = MeanShift(bandwidth = bandwidth2, bin_seeding=True)
            ms2.fit(flatimg)
            labels2=ms2.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels2)
            labels2 = np.reshape(labels2, (756,1008))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented2 = color.label2rgb(labels2, img, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_2_segmentationmap.png", segmented2)

            ### boundary overlay
            boundary2 = mark_boundaries(img, labels2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            g2 = graph.rag_mean_color(img, labels2, mode='similarity')
            nx.draw(g2)
            plt.savefig(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 4, third parameter set is in bandwith of the mean shift algorithm: quantile = 0.1 and number of samples = 700
            bandwidth3 = estimate_bandwidth(flatimg, quantile=0.1, n_samples=700)
            ### apply mean shift segmentation to image 4 with the third parameter set
            ms3 = MeanShift(bandwidth = bandwidth3, bin_seeding=True)
            ms3.fit(flatimg)
            labels3=ms3.labels_  
            # cluster_centers = ms.cluster_centers_
            # labels_unique = np.unique(labels3)
            labels3 = np.reshape(labels3, (756,1008))
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)

            ### segmentation map
            segmented3 = color.label2rgb(labels3, img, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_3_segmentationmap.png", segmented3)

            ### boundary overlay
            boundary3 = mark_boundaries(img, labels3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            g3 = graph.rag_mean_color(img, labels3, mode='similarity')
            nx.draw(g3)
            plt.savefig(OUTPUT_PATH + "B4_algorithm_meanshift_parameterset_3_rag.png")
            plt.clf() 

        ### for n-cut segmentation
        elif t == "ncut":
            io.imsave(OUTPUT_PATH + "B4_algorithm_ncut_original.png", image)
            ### for image 4, first parameter set is in labels of the n-cut algorithm: compactness = 20 and number of segments = 300
            segments = segmentation.slic(image, compactness=20, n_segments=300)
            g = graph.rag_mean_color(image, segments, mode='similarity')
            ### apply n-cut segmentation to image 4 with the first parameter set 
            graph_cut_segments = graph.cut_normalized(segments, g)
            # labels_unique = np.unique(graph_cut_segments)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented = color.label2rgb(graph_cut_segments, image, kind='avg')
            segmented = img_as_ubyte(segmented)
            io.imsave(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_1_segmentationmap.png", segmented)
            
            ### boundary overlay
            boundary = segmentation.mark_boundaries(image, graph_cut_segments, (0, 0, 0), mode='thick')
            boundary = img_as_ubyte(boundary)
            io.imsave(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_1_boundaryoverlay.png", boundary)

            ### region adjacency graph
            rag = graph.rag_mean_color(image, graph_cut_segments, mode='similarity')
            nx.draw(rag)
            plt.savefig(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_1_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 4, second parameter set is in labels of the n-cut algorithm: compactness = 10 and number of segments = 300
            segments2 = segmentation.slic(image, compactness=10, n_segments=300)
            g2 = graph.rag_mean_color(image, segments2, mode='similarity')
            ### apply n-cut segmentation to image 4 with the second parameter set 
            graph_cut_segments2 = graph.cut_normalized(segments2, g2)
            # labels_unique = np.unique(graph_cut_segments2)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented2 = color.label2rgb(graph_cut_segments2, image, kind='avg')
            segmented2 = img_as_ubyte(segmented2)
            io.imsave(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_2_segmentationmap.png", segmented2)
            
            ### boundary overlay
            boundary2 = segmentation.mark_boundaries(image, graph_cut_segments2, (0, 0, 0), mode='thick')
            boundary2 = img_as_ubyte(boundary2)
            io.imsave(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_2_boundaryoverlay.png", boundary2)

            ### region adjacency graph
            rag2 = graph.rag_mean_color(image, graph_cut_segments2, mode='similarity')
            nx.draw(rag2)
            plt.savefig(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_2_rag.png")
            plt.clf()

            ##############################################################################################################################

            ### for image 4, third parameter set is in labels of the n-cut algorithm: compactness = 10 and number of segments = 500
            segments3 = segmentation.slic(image, compactness=10, n_segments=500)
            g3 = graph.rag_mean_color(image, segments3, mode='similarity')
            ### apply n-cut segmentation to image 4 with the third parameter set 
            graph_cut_segments3 = graph.cut_normalized(segments3, g3)
            # labels_unique = np.unique(graph_cut_segments3)
            # n_clusters_ = len(labels_unique)
            # print("number of estimated clusters : %d" % n_clusters_)
            
            ### segmentation map
            segmented3 = color.label2rgb(graph_cut_segments3, image, kind='avg')
            segmented3 = img_as_ubyte(segmented3)
            io.imsave(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_3_segmentationmap.png", segmented3)
            
            ### boundary overlay
            boundary3 = segmentation.mark_boundaries(image, graph_cut_segments3, (0, 0, 0), mode='thick')
            boundary3 = img_as_ubyte(boundary3)
            io.imsave(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_3_boundaryoverlay.png", boundary3)

            ### region adjacency graph
            rag3 = graph.rag_mean_color(image, graph_cut_segments3, mode='similarity')
            nx.draw(rag3)
            plt.savefig(OUTPUT_PATH + "B4_algorithm_ncut_parameterset_3_rag.png")
            plt.clf()

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    object_counting(1)
    object_counting(2)
    object_counting(3)

    clustering(1, "meanshift")
    clustering(1, "ncut")

    clustering(2, "meanshift")
    clustering(2, "ncut")

    clustering(3, "meanshift")
    clustering(3, "ncut")

    clustering(4, "meanshift")
    clustering(4, "ncut")

    ### sometimes, algorithms could not cluster the image and creating region adjacency graph for non clustered image gives
    ### error and terminates the program. so, if you encounter that situation, try rerun the program. also, when rerunning, you 
    ### can comment out the corresponding region adjacency graph creations depending on the error line. please look at the lines
    ### for possible error creations --> (167-171), (197-201), (227-231), (255-259), (282-286), (309-313), (347-351), (377-381),
    ### (407-411), (435-439), (462-466), (489-493), (525-529), (555-559), (585-589), (613-617), (640-644), (667-671), (705-709),
    ### (735-739), (765-769), (793-797), (820-824), (847-851).